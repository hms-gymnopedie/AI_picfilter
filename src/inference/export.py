"""모델 export 유틸리티.

학습된 모델(NILUT, ImageAdaptive3DLUT)을 .cube, ONNX 포맷으로 변환한다.
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from src.models.nilut import NILUT
from src.data.cube_parser import CubeParser

logger = logging.getLogger(__name__)


def _detect_model_type(checkpoint: dict) -> str:
    """체크포인트 state_dict 키로 모델 타입 자동 감지."""
    keys = list(checkpoint.get("model_state_dict", {}).keys())
    if any(k.startswith("basis_luts") for k in keys):
        return "lut3d"
    return "nilut"


def _load_lut3d_from_checkpoint(
    model_path: str | Path,
    device: str = "cpu",
):
    """체크포인트에서 ImageAdaptive3DLUT 모델 복원."""
    from src.models.lut3d import ImageAdaptive3DLUT

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없음: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint.get("model_config", {})

    model = ImageAdaptive3DLUT(
        n_basis=config.get("n_basis", 3),
        lut_size=config.get("lut_size", 33),
        backbone_input_size=config.get("backbone_input_size", 256),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _load_nilut_from_checkpoint(
    model_path: str | Path,
    device: str = "cpu",
) -> NILUT:
    """체크포인트에서 NILUT 모델 복원.

    체크포인트의 model_config를 사용하여 모델 구조를 자동으로 추론한다.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없음: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint.get("model_config", {})

    # state_dict 분석으로 hidden_dims 역추론
    state_dict = checkpoint["model_state_dict"]
    hidden_dims = _infer_hidden_dims(state_dict)

    model = NILUT(
        hidden_dims=hidden_dims,
        num_styles=config.get("num_styles"),
        style_dim=config.get("style_dim", 64),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _infer_hidden_dims(state_dict: dict) -> list[int]:
    """state_dict에서 hidden_dims 역추론."""
    hidden_dims = []
    for key, tensor in state_dict.items():
        if key.startswith("mlp.") and key.endswith(".weight"):
            # mlp.0.weight, mlp.2.weight, ... 순서
            out_dim = tensor.shape[0]
            # 마지막 Linear(출력 3) 제외
            if out_dim != 3:
                hidden_dims.append(out_dim)
    return hidden_dims if hidden_dims else [256, 256, 256]


def export_to_cube(
    model_path: str | Path,
    output_path: str | Path,
    cube_size: int = 33,
    style_idx: int = 0,
    device: str = "cpu",
    reference_image: Optional[np.ndarray] = None,
) -> None:
    """학습된 모델을 .cube 파일로 변환.

    NILUT와 ImageAdaptive3DLUT 모두 지원한다.
    Photoshop, DaVinci Resolve 등과 호환된다.

    Args:
        model_path: 학습된 모델(.pt) 경로
        output_path: 출력 .cube 파일 경로
        cube_size: LUT 그리드 크기 (17, 33, 64)
        style_idx: 변환할 스타일 인덱스 (NILUT 전용)
        device: 추론 디바이스
        reference_image: 레퍼런스 이미지 [H, W, 3], float32 (lut3d 전용).
            None이면 중간값(0.5) 이미지 사용.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없음: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model_type = _detect_model_type(checkpoint)

    if model_type == "lut3d":
        _export_lut3d_to_cube(
            model_path, output_path, cube_size, device, reference_image
        )
    else:
        _export_nilut_to_cube(model_path, output_path, cube_size, style_idx, device)


def _export_nilut_to_cube(
    model_path: str | Path,
    output_path: str | Path,
    cube_size: int,
    style_idx: int,
    device: str,
) -> None:
    """NILUT 모델을 .cube로 변환 (내부 함수)."""
    model = _load_nilut_from_checkpoint(model_path, device)

    coords = torch.linspace(0.0, 1.0, cube_size, device=device)
    r, g, b = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid = torch.stack([r.ravel(), g.ravel(), b.ravel()], dim=-1)

    style_tensor: Optional[torch.Tensor] = None
    if model.num_styles is not None:
        style_tensor = torch.full(
            (grid.shape[0],), fill_value=style_idx, dtype=torch.long, device=device
        )

    with torch.no_grad():
        out = model(grid, style_idx=style_tensor)

    lut_np = out.cpu().numpy().reshape(cube_size, cube_size, cube_size, 3)
    lut_np = np.clip(lut_np, 0.0, 1.0).astype(np.float32)

    parser = CubeParser()
    parser.write(lut_np, output_path, title=f"AI_picfilter NILUT Style {style_idx}")
    logger.info(f".cube export 완료 (NILUT): {output_path} (size={cube_size})")


def _export_lut3d_to_cube(
    model_path: str | Path,
    output_path: str | Path,
    cube_size: int,
    device: str,
    reference_image: Optional[np.ndarray],
) -> None:
    """ImageAdaptive3DLUT 모델을 .cube로 변환 (내부 함수).

    레퍼런스 이미지로 backbone을 실행하여 blended LUT를 추출한다.
    lut_size != cube_size이면 보간하여 크기를 맞춘다.
    """
    import torch.nn.functional as F

    model = _load_lut3d_from_checkpoint(model_path, device)

    # 레퍼런스 이미지 준비
    if reference_image is not None:
        img_np = np.asarray(reference_image, dtype=np.float32)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    else:
        # 중간값(0.5) 이미지 — 평균적인 LUT 반환
        img_t = torch.full((1, 3, 256, 256), 0.5, device=device)

    with torch.no_grad():
        # blended LUT: [lut_size, lut_size, lut_size, 3]
        blended = model.get_blended_lut(img_t)

    lut_np = blended.cpu().numpy().astype(np.float32)

    # cube_size가 모델 lut_size와 다르면 trilinear 보간
    if lut_np.shape[0] != cube_size:
        lut_t = torch.from_numpy(lut_np).permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, S, S, S]
        lut_t = F.interpolate(lut_t, size=(cube_size, cube_size, cube_size), mode="trilinear",
                              align_corners=True)
        lut_np = lut_t.squeeze(0).permute(1, 2, 3, 0).numpy()

    lut_np = np.clip(lut_np, 0.0, 1.0).astype(np.float32)

    parser = CubeParser()
    parser.write(lut_np, output_path, title="AI_picfilter Image-Adaptive 3D LUT")
    logger.info(f".cube export 완료 (lut3d): {output_path} (size={cube_size})")


def export_to_onnx(
    model_path: str | Path,
    output_path: str | Path,
    opset_version: int = 17,
    style_idx: int = 0,
) -> None:
    """ONNX 포맷으로 export.

    Args:
        model_path: 학습된 모델 경로
        output_path: 출력 .onnx 파일 경로
        opset_version: ONNX opset 버전
        style_idx: export할 스타일 인덱스 (다중 스타일 모드에서 단일 스타일로 고정)
    """
    model = _load_nilut_from_checkpoint(model_path, device="cpu")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 더미 입력 [1, 3]
    dummy_input = torch.rand(1, 3)

    # 다중 스타일인 경우 스타일 인덱스 고정
    if model.num_styles is not None:

        class NILUTWithStyle(torch.nn.Module):
            def __init__(self, inner: NILUT, sid: int):
                super().__init__()
                self.inner = inner
                self.style_idx_val = sid

            def forward(self, rgb: torch.Tensor) -> torch.Tensor:
                b = rgb.shape[0]
                sidx = torch.full((b,), self.style_idx_val, dtype=torch.long)
                return self.inner(rgb, style_idx=sidx)

        export_model = NILUTWithStyle(model, style_idx)
        input_names = ["rgb"]
    else:
        export_model = model
        input_names = ["rgb"]

    export_model.eval()
    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=["rgb_out"],
        dynamic_axes={"rgb": {0: "batch_size"}, "rgb_out": {0: "batch_size"}},
    )

    logger.info(f"ONNX export 완료: {output_path} (opset={opset_version})")
