"""이미지에 필터 적용.

학습된 NILUT/ImageAdaptive3DLUT 모델 또는 .cube 파일을 사용하여
이미지에 색상 필터를 적용한다.
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def apply_filter(
    image: np.ndarray,
    model_path: Optional[str | Path] = None,
    cube_path: Optional[str | Path] = None,
    style_idx: int = 0,
    device: str = "cpu",
    intensity: float = 1.0,
) -> np.ndarray:
    """이미지에 필터 적용.

    model_path 또는 cube_path 중 하나를 반드시 지정해야 한다.
    모델 타입(NILUT vs ImageAdaptive3DLUT)은 체크포인트에서 자동 감지한다.

    Args:
        image: 입력 이미지 [H, W, 3], uint8 또는 float32
        model_path: 학습된 모델(.pt) 경로 (NILUT 또는 lut3d)
        cube_path: .cube 파일 경로
        style_idx: 적용할 스타일 인덱스 (다중 스타일 NILUT에서만 사용)
        device: 추론 디바이스 ('cpu', 'cuda', 'mps')
        intensity: 필터 강도 [0.0, 1.0]. 1.0이면 원본 필터, 0.0이면 입력 이미지

    Returns:
        필터 적용된 이미지 [H, W, 3], 입력과 동일한 dtype

    Raises:
        ValueError: model_path와 cube_path 모두 None이거나 모두 지정된 경우
    """
    if model_path is None and cube_path is None:
        raise ValueError("model_path 또는 cube_path 중 하나를 지정해야 합니다.")
    if model_path is not None and cube_path is not None:
        raise ValueError("model_path와 cube_path를 동시에 지정할 수 없습니다.")

    # 입력 dtype 기록 후 float32 [0, 1]로 정규화
    original_dtype = image.dtype
    image_f = image.astype(np.float32)
    if original_dtype == np.uint8:
        image_f = image_f / 255.0

    image_f = np.clip(image_f, 0.0, 1.0)

    if model_path is not None:
        result_f = _apply_model(image_f, model_path, style_idx, device)
    else:
        result_f = _apply_cube(image_f, cube_path)

    # intensity 블렌딩: result = input * (1 - intensity) + output * intensity
    if intensity < 1.0:
        result_f = image_f * (1.0 - intensity) + result_f * intensity

    result_f = np.clip(result_f, 0.0, 1.0)

    # 원래 dtype으로 복원
    if original_dtype == np.uint8:
        return (result_f * 255.0).round().astype(np.uint8)
    return result_f.astype(np.float32)


def _apply_model(
    image_f: np.ndarray,
    model_path: str | Path,
    style_idx: int,
    device: str,
) -> np.ndarray:
    """체크포인트 타입을 자동 감지하여 적절한 모델로 필터 적용 (내부 함수)."""
    from src.inference.export import _detect_model_type
    import torch as _torch

    path = Path(model_path)
    checkpoint = _torch.load(path, map_location=device, weights_only=True)
    model_type = _detect_model_type(checkpoint)

    if model_type == "lut3d":
        return _apply_lut3d_model(image_f, model_path, device)
    else:
        return _apply_nilut_model(image_f, model_path, style_idx, device)


def _apply_nilut_model(
    image_f: np.ndarray,
    model_path: str | Path,
    style_idx: int,
    device: str,
) -> np.ndarray:
    """NILUT 모델을 사용한 필터 적용 (내부 함수)."""
    from src.inference.export import _load_nilut_from_checkpoint

    model = _load_nilut_from_checkpoint(model_path, device=device)

    h, w, _ = image_f.shape
    pixels = torch.from_numpy(image_f.reshape(-1, 3)).to(device)  # [N, 3]

    style_tensor: Optional[torch.Tensor] = None
    if model.num_styles is not None:
        style_tensor = torch.full(
            (pixels.shape[0],), fill_value=style_idx, dtype=torch.long, device=device
        )

    with torch.no_grad():
        out = model(pixels, style_idx=style_tensor)  # [N, 3]

    return out.cpu().numpy().reshape(h, w, 3)


def _apply_lut3d_model(
    image_f: np.ndarray,
    model_path: str | Path,
    device: str,
) -> np.ndarray:
    """ImageAdaptive3DLUT 모델을 사용한 필터 적용 (내부 함수).

    이미지 전체를 [B, 3, H, W] 텐서로 변환하여 forward pass 수행.
    backbone이 이미지 내용에 따라 basis LUT 가중치를 동적으로 결정한다.
    """
    from src.inference.export import _load_lut3d_from_checkpoint

    model = _load_lut3d_from_checkpoint(model_path, device=device)

    h, w, _ = image_f.shape
    # [H, W, 3] -> [1, 3, H, W]
    img_t = torch.from_numpy(image_f).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out_t = model(img_t)  # [1, 3, H, W]

    # [1, 3, H, W] -> [H, W, 3]
    return out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()


def _apply_cube(image_f: np.ndarray, cube_path: str | Path) -> np.ndarray:
    """3D LUT (.cube)를 사용한 필터 적용 (내부 함수)."""
    from src.data.cube_parser import CubeParser
    from src.utils.lut import apply_lut

    parser = CubeParser()
    lut = parser.read(cube_path)
    return apply_lut(image_f, lut)


def apply_filter_batch(
    images: list[np.ndarray],
    model_path: str | Path,
    style_idx: int = 0,
    device: str = "cpu",
    batch_size: int = 4,
) -> list[np.ndarray]:
    """여러 이미지에 배치로 필터 적용.

    모델을 한 번만 로드하고 배치 단위로 처리하여 효율적이다.
    모델 타입(NILUT vs lut3d)은 체크포인트에서 자동 감지한다.

    Args:
        images: 입력 이미지 리스트, 각 [H, W, 3]
        model_path: 학습된 모델 경로
        style_idx: 적용할 스타일 인덱스 (NILUT 다중 스타일 전용)
        device: 추론 디바이스
        batch_size: 동시에 처리할 이미지 수 (메모리에 맞게 조절)

    Returns:
        필터 적용된 이미지 리스트
    """
    from src.inference.export import (
        _detect_model_type,
        _load_nilut_from_checkpoint,
        _load_lut3d_from_checkpoint,
    )

    path = Path(model_path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model_type = _detect_model_type(checkpoint)

    if model_type == "lut3d":
        model = _load_lut3d_from_checkpoint(model_path, device=device)
    else:
        model = _load_nilut_from_checkpoint(model_path, device=device)

    results = []

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        batch_results = []

        for img in batch_imgs:
            original_dtype = img.dtype
            img_f = img.astype(np.float32)
            if original_dtype == np.uint8:
                img_f = img_f / 255.0
            img_f = np.clip(img_f, 0.0, 1.0)

            h, w, _ = img_f.shape

            with torch.no_grad():
                if model_type == "lut3d":
                    img_t = (
                        torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)
                    )
                    out_t = model(img_t)  # [1, 3, H, W]
                    result_f = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                else:
                    pixels = torch.from_numpy(img_f.reshape(-1, 3)).to(device)
                    style_tensor: Optional[torch.Tensor] = None
                    if model.num_styles is not None:
                        style_tensor = torch.full(
                            (pixels.shape[0],),
                            fill_value=style_idx,
                            dtype=torch.long,
                            device=device,
                        )
                    out = model(pixels, style_idx=style_tensor)
                    result_f = out.cpu().numpy().reshape(h, w, 3)

            result_f = np.clip(result_f, 0.0, 1.0)

            if original_dtype == np.uint8:
                result_f = (result_f * 255.0).round().astype(np.uint8)
            else:
                result_f = result_f.astype(np.float32)

            batch_results.append(result_f)

        results.extend(batch_results)
        logger.info(f"배치 {i // batch_size + 1} 처리 완료 ({len(batch_results)}장)")

    return results
