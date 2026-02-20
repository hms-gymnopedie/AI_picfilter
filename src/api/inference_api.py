"""ML 추론 인터페이스.

백엔드 Celery 워커가 호출하는 ML 함수 모음.
NILUT와 ImageAdaptive3DLUT 두 모델 타입을 지원한다.

지원 model_type 값:
    "nilut"          — Neural Implicit LUT (기본값, Phase 1)
    "adaptive_3dlut" — Image-Adaptive 3D LUT (Phase 2)
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# 백엔드 호환용 progress_callback 타입: (current: float, total: float) -> None
ProgressCallback = Optional[Callable[[float, float], None]]


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _resolve_device() -> str:
    """사용 가능한 최적 디바이스 선택."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _lut3d_trainer_fit(
    reference_images: list[str],
    output_path: str,
    config: Dict[str, Any],
    progress_callback: ProgressCallback,
) -> Dict[str, Any]:
    """ImageAdaptive3DLUT 모델 학습 실행."""
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    from PIL import Image as PILImage

    from src.models.lut3d import ImageAdaptive3DLUT
    from src.training.losses import LUT3DCombinedLoss
    from src.training.schedulers import get_scheduler

    device = _resolve_device()
    logger.info(f"lut3d 학습 시작 (device={device}, images={len(reference_images)})")

    # 설정 파싱
    n_basis = int(config.get("n_basis", 3))
    lut_size = int(config.get("lut_size", 33))
    backbone_input_size = int(config.get("backbone_input_size", 256))
    epochs = int(config.get("epochs", 100))
    lr = float(config.get("lr", 1e-4))
    lambda_perceptual = float(config.get("lambda_perceptual", 0.1))
    lambda_monotonic = float(config.get("lambda_monotonic", 10.0))
    use_perceptual = bool(config.get("use_perceptual", True))
    image_size = config.get("image_size", [256, 256])

    # 레퍼런스 이미지가 없으면 에러
    if not reference_images:
        raise ValueError("reference_images가 비어 있습니다.")

    # -----------------------------------------------------------------------
    # 단순 single-image self-supervised 데이터셋
    # reference_image를 augmentation으로 변환한 쌍(자기 자신)으로 학습
    # 실제 환경에서는 paired 데이터셋을 사용해야 한다
    # -----------------------------------------------------------------------
    class _RefImageDataset(Dataset):
        def __init__(self, paths: list[str], size: tuple[int, int]) -> None:
            self.paths = paths
            self.size = size

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int):
            img = PILImage.open(self.paths[idx]).convert("RGB")
            img = img.resize(self.size[::-1], PILImage.BILINEAR)  # (W, H) 순서
            arr = np.array(img, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
            return t, t  # (input, target) — identity target

    dataset = _RefImageDataset(reference_images, tuple(image_size))
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=True, num_workers=0)

    # 모델 및 옵티마이저
    model = ImageAdaptive3DLUT(
        n_basis=n_basis,
        lut_size=lut_size,
        backbone_input_size=backbone_input_size,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = get_scheduler(optimizer, "cosine", epochs, warmup_epochs=min(5, epochs // 10))
    criterion = LUT3DCombinedLoss(
        lambda_perceptual=lambda_perceptual,
        lambda_monotonic=lambda_monotonic,
        use_perceptual=use_perceptual,
    )

    best_loss = float("inf")
    history: list[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            pred = model(inp)
            loss_dict = criterion(pred, tgt, model.basis_luts)
            loss = loss_dict["total"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss})
        logger.debug(f"epoch {epoch}/{epochs} loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

        if progress_callback is not None:
            progress_callback(float(epoch), float(epochs))

    # 체크포인트 저장
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "n_basis": n_basis,
                "lut_size": lut_size,
                "backbone_input_size": backbone_input_size,
            },
            "training_history": history,
        },
        out_path,
    )
    logger.info(f"lut3d 모델 저장 완료: {out_path}")
    return {"best_loss": best_loss, "epochs_trained": epochs}


def _nilut_trainer_fit(
    reference_images: list[str],
    output_path: str,
    config: Dict[str, Any],
    progress_callback: ProgressCallback,
) -> Dict[str, Any]:
    """NILUT 모델 학습 실행."""
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import numpy as np
    from PIL import Image as PILImage

    from src.models.nilut import NILUT
    from src.training.losses import CombinedLoss
    from src.training.schedulers import get_scheduler

    device = _resolve_device()
    logger.info(f"NILUT 학습 시작 (device={device}, images={len(reference_images)})")

    hidden_dims = config.get("hidden_dims", [256, 256, 256])
    epochs = int(config.get("epochs", 100))
    lr = float(config.get("lr", 1e-4))
    lambda_delta_e = float(config.get("lambda_delta_e", 0.5))
    lambda_smooth = float(config.get("lambda_smooth", 0.01))

    if not reference_images:
        raise ValueError("reference_images가 비어 있습니다.")

    # 레퍼런스 이미지에서 픽셀 쌍 추출
    all_pixels: list = []
    for p in reference_images:
        img = PILImage.open(p).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        all_pixels.append(flat)

    pixels_np = np.concatenate(all_pixels, axis=0)
    # identity target (self-supervised)
    inp_t = torch.from_numpy(pixels_np)
    tgt_t = torch.from_numpy(pixels_np)

    dataset = torch.utils.data.TensorDataset(inp_t, tgt_t)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=0)

    model = NILUT(hidden_dims=hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = get_scheduler(optimizer, "cosine", epochs, warmup_epochs=min(5, epochs // 10))
    criterion = CombinedLoss(
        lambda_delta_e=lambda_delta_e,
        lambda_smooth=lambda_smooth,
    )

    best_loss = float("inf")
    history: list[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            pred = model(inp)
            loss_dict = criterion(pred, tgt, model)
            loss = loss_dict["total"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss

        if progress_callback is not None:
            progress_callback(float(epoch), float(epochs))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "hidden_dims": hidden_dims,
                "num_styles": None,
                "style_dim": 64,
            },
            "training_history": history,
        },
        out_path,
    )
    logger.info(f"NILUT 모델 저장 완료: {out_path}")
    return {"best_loss": best_loss, "epochs_trained": epochs}


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def run_style_learning(
    reference_images: list[str],
    output_path: str,
    model_type: str,
    config: Dict[str, Any],
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    """레퍼런스 이미지로부터 스타일 학습.

    Args:
        reference_images: 레퍼런스 이미지 파일 경로 목록
        output_path: 학습된 모델 저장 경로 (.pt 파일)
        model_type: 모델 타입
            - "nilut"          : Neural Implicit LUT (기본)
            - "adaptive_3dlut" : Image-Adaptive 3D LUT
        config: 학습 설정 딕셔너리.
            공통: epochs, lr
            NILUT: hidden_dims, lambda_delta_e, lambda_smooth
            adaptive_3dlut: n_basis, lut_size, backbone_input_size,
                            lambda_perceptual, lambda_monotonic, use_perceptual
        progress_callback: 진행 콜백 (current, total)

    Returns:
        {
            "status": "success",
            "model_path": str,
            "model_type": str,
            "config": dict,
        }
    """
    try:
        if model_type == "adaptive_3dlut":
            stats = _lut3d_trainer_fit(reference_images, output_path, config, progress_callback)
        elif model_type in ("nilut", "dlut"):
            # dlut는 현재 nilut로 대체
            if model_type == "dlut":
                logger.warning("dlut 모델은 현재 nilut로 대체됩니다.")
            stats = _nilut_trainer_fit(reference_images, output_path, config, progress_callback)
        else:
            raise ValueError(f"지원하지 않는 model_type: {model_type!r}")

        return {
            "status": "success",
            "model_path": str(output_path),
            "model_type": model_type,
            "config": {**config, **stats},
        }

    except Exception as exc:
        logger.exception("run_style_learning 실패")
        return {
            "status": "error",
            "error": str(exc),
            "model_path": str(output_path),
            "model_type": model_type,
            "config": config,
        }


def run_apply_filter(
    input_image: str,
    model_path: str,
    output_path: str,
    intensity: float = 1.0,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    """학습된 스타일을 대상 이미지에 적용.

    모델 타입(NILUT vs lut3d)은 체크포인트에서 자동 감지한다.

    Args:
        input_image: 대상 이미지 파일 경로
        model_path: 학습된 모델 경로 (.pt 파일)
        output_path: 결과 이미지 저장 경로
        intensity: 필터 강도 (0.0 ~ 1.0)
        progress_callback: 진행 콜백 (current, total)

    Returns:
        {
            "status": "success",
            "output_image": str,
            "processing_time_ms": float,
        }
    """
    try:
        import numpy as np
        from PIL import Image as PILImage
        from src.inference.apply_filter import apply_filter

        if progress_callback is not None:
            progress_callback(0.0, 1.0)

        img_np = np.array(PILImage.open(input_image).convert("RGB"), dtype=np.uint8)

        t0 = time.perf_counter()
        result_np = apply_filter(
            image=img_np,
            model_path=model_path,
            intensity=intensity,
            device=_resolve_device(),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(result_np).save(str(out_path))

        if progress_callback is not None:
            progress_callback(1.0, 1.0)

        logger.info(f"필터 적용 완료: {out_path} ({elapsed_ms:.1f}ms)")

        return {
            "status": "success",
            "output_image": str(out_path),
            "processing_time_ms": elapsed_ms,
        }

    except Exception as exc:
        logger.exception("run_apply_filter 실패")
        return {
            "status": "error",
            "error": str(exc),
            "output_image": str(output_path),
            "processing_time_ms": 0.0,
        }


def run_export_cube(
    model_path: str,
    output_path: str,
    lut_size: int = 33,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    """.cube LUT 파일로 내보내기.

    NILUT와 ImageAdaptive3DLUT 모두 지원한다.

    Args:
        model_path: 학습된 모델 경로 (.pt 파일)
        output_path: .cube 파일 저장 경로
        lut_size: LUT 해상도 (17, 33, 65)
        progress_callback: 진행 콜백 (current, total)

    Returns:
        {
            "status": "success",
            "cube_file": str,
            "lut_size": int,
            "file_size_bytes": int,
        }
    """
    try:
        from src.inference.export import export_to_cube

        if progress_callback is not None:
            progress_callback(0.0, 1.0)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        export_to_cube(
            model_path=model_path,
            output_path=str(out_path),
            cube_size=lut_size,
            device=_resolve_device(),
        )

        file_size = out_path.stat().st_size

        if progress_callback is not None:
            progress_callback(1.0, 1.0)

        logger.info(f".cube export 완료: {out_path} ({file_size} bytes)")

        return {
            "status": "success",
            "cube_file": str(out_path),
            "lut_size": lut_size,
            "file_size_bytes": file_size,
        }

    except Exception as exc:
        logger.exception("run_export_cube 실패")
        return {
            "status": "error",
            "error": str(exc),
            "cube_file": str(output_path),
            "lut_size": lut_size,
            "file_size_bytes": 0,
        }


def get_model_info(model_path: str) -> Dict[str, Any]:
    """저장된 모델의 메타데이터 반환.

    Args:
        model_path: 모델 파일 경로 (.pt)

    Returns:
        {
            "model_type": str,       # "nilut" 또는 "lut3d"
            "model_config": dict,    # 모델 구성 하이퍼파라미터
            "file_size_bytes": int,
            "param_count": int,      # 총 파라미터 수
        }
    """
    import torch
    from src.inference.export import _detect_model_type

    path = Path(model_path)
    if not path.exists():
        return {
            "model_type": "unknown",
            "model_config": {},
            "file_size_bytes": 0,
            "param_count": 0,
            "error": f"파일 없음: {path}",
        }

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model_type = _detect_model_type(checkpoint)
    model_config = checkpoint.get("model_config", {})
    file_size = path.stat().st_size

    # 파라미터 수 계산
    state_dict = checkpoint.get("model_state_dict", {})
    param_count = sum(t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor))

    return {
        "model_type": model_type,
        "model_config": model_config,
        "file_size_bytes": file_size,
        "param_count": param_count,
    }
