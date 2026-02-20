"""Hald CLUT 이미지 생성 및 처리.

Hald 이미지는 3D LUT를 2D 이미지로 인코딩한 것이다.
레퍼런스 이미지에 적용된 보정을 역추출하는 데 사용한다.

Hald level L:
  - LUT 크기 = L^2  (예: L=8 -> 64^3 LUT)
  - 이미지 크기 = (L^3) x (L^3) 픽셀
"""

import numpy as np
from pathlib import Path

try:
    from PIL import Image as PILImage

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def generate_hald_image(level: int = 8) -> np.ndarray:
    """Hald CLUT 이미지 생성 (항등 변환).

    Args:
        level: Hald 레벨. LUT 크기 = level^2, 이미지 크기 = level^3 x level^3

    Returns:
        Hald 이미지 [H, W, 3], uint8
    """
    lut_size = level**2  # e.g., 64
    img_size = level**3  # e.g., 512

    # LUT 좌표 생성 (B-major 순서)
    coords = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
    r, g, b = np.meshgrid(coords, coords, coords, indexing="ij")
    # [lut_size^3, 3] B-major: B 가장 빠름
    lut_flat = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=-1)

    # [img_size, img_size, 3] 로 reshape
    hald = (lut_flat.reshape(img_size, img_size, 3) * 255.0).round().astype(np.uint8)
    return hald


def extract_lut_from_hald(
    original_hald: np.ndarray,
    processed_hald: np.ndarray,
    level: int = 8,
) -> np.ndarray:
    """보정이 적용된 Hald 이미지에서 3D LUT 추출.

    original_hald의 각 픽셀이 기준 색상을 나타내고,
    processed_hald의 동일 위치 픽셀이 변환 후 색상을 나타낸다.

    Args:
        original_hald: 원본 Hald 이미지 [H, W, 3], uint8
        processed_hald: 보정이 적용된 Hald 이미지 [H, W, 3], uint8
        level: Hald 레벨

    Returns:
        3D LUT [size, size, size, 3], float32, 범위 [0, 1]
    """
    lut_size = level**2
    img_size = level**3

    processed = np.asarray(processed_hald, dtype=np.float32) / 255.0
    # [img_size, img_size, 3] -> [lut_size^3, 3] -> [lut_size, lut_size, lut_size, 3]
    lut_flat = processed.reshape(img_size * img_size, 3)

    if lut_flat.shape[0] != lut_size**3:
        raise ValueError(
            f"Hald 이미지 크기 불일치: 예상 {lut_size**3} 픽셀, 실제 {lut_flat.shape[0]}"
        )

    lut = lut_flat.reshape(lut_size, lut_size, lut_size, 3).astype(np.float32)
    return lut.clip(0.0, 1.0)


def save_hald_image(hald: np.ndarray, path: str | Path) -> None:
    """Hald 이미지를 파일로 저장.

    Args:
        hald: Hald 이미지 배열 [H, W, 3], uint8
        path: 출력 파일 경로 (.png 권장)

    Raises:
        ImportError: Pillow가 설치되어 있지 않을 때
    """
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow가 필요합니다: pip install Pillow")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hald_uint8 = np.asarray(hald, dtype=np.uint8)
    img = PILImage.fromarray(hald_uint8, mode="RGB")
    img.save(path)
