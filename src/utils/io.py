"""이미지 I/O 헬퍼."""

import numpy as np
from pathlib import Path
from typing import Optional

try:
    from PIL import Image as PILImage

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def load_image(
    path: str | Path,
    size: Optional[tuple[int, int]] = None,
    as_float: bool = True,
) -> np.ndarray:
    """이미지 파일 로드.

    Args:
        path: 이미지 파일 경로
        size: 리사이즈 크기 (H, W). None이면 원본 크기 유지
        as_float: True이면 [0, 1] float32, False이면 [0, 255] uint8

    Returns:
        이미지 배열 [H, W, 3] RGB

    Raises:
        ImportError: Pillow가 설치되어 있지 않을 때
        FileNotFoundError: 파일이 존재하지 않을 때
    """
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow가 필요합니다: pip install Pillow")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없음: {path}")

    img = PILImage.open(path).convert("RGB")  # 항상 RGB로 변환

    if size is not None:
        h, w = size
        img = img.resize((w, h), PILImage.BICUBIC)

    arr = np.asarray(img)  # [H, W, 3], uint8

    if as_float:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.uint8)


def save_image(image: np.ndarray, path: str | Path) -> None:
    """이미지 파일 저장.

    Args:
        image: 이미지 배열 [H, W, 3], float32 [0, 1] 또는 uint8 [0, 255]
        path: 출력 파일 경로

    Raises:
        ImportError: Pillow가 설치되어 있지 않을 때
    """
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow가 필요합니다: pip install Pillow")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        # float32 -> uint8
        arr = np.clip(image, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    else:
        arr = image

    img = PILImage.fromarray(arr, mode="RGB")
    img.save(path)
