"""이미지 변환 및 전처리.

학습/추론에 필요한 이미지 전처리 변환 함수 모음.
color.py의 변환 함수를 래핑하여 transforms 레이어를 제공한다.
"""

import torch
import numpy as np
from typing import Optional

from src.utils.color import (
    _srgb_to_linear_torch,
    rgb_to_lab as _rgb_to_lab_impl,
)


def normalize(
    image: torch.Tensor,
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> torch.Tensor:
    """이미지 정규화.

    Args:
        image: [C, H, W] 또는 [B, C, H, W] 텐서, 범위 [0, 1]
        mean: 채널별 평균 리스트. None이면 [0, 1] 그대로 반환
        std: 채널별 표준편차 리스트. None이면 [0, 1] 그대로 반환

    Returns:
        정규화된 텐서 (ImageNet 스타일 mean/std 지정 시)
    """
    if mean is None or std is None:
        return image.clamp(0.0, 1.0)

    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device)

    # [C] -> 브로드캐스트 가능 shape
    if image.ndim == 3:
        mean_t = mean_t.view(-1, 1, 1)
        std_t = std_t.view(-1, 1, 1)
    elif image.ndim == 4:
        mean_t = mean_t.view(1, -1, 1, 1)
        std_t = std_t.view(1, -1, 1, 1)

    return (image - mean_t) / std_t.clamp(min=1e-8)


def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    """sRGB -> Linear RGB 변환 (감마 디코딩).

    Args:
        image: sRGB 이미지 텐서, 범위 [0, 1]

    Returns:
        Linear RGB 텐서, 범위 [0, 1]
    """
    return _srgb_to_linear_torch(image)


def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    """Linear RGB -> sRGB 변환 (감마 인코딩).

    Args:
        image: Linear RGB 텐서, 범위 [0, 1]

    Returns:
        sRGB 이미지 텐서, 범위 [0, 1]
    """
    x = image.clamp(0.0, 1.0)
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * x.pow(1.0 / 2.4) - 0.055,
    )


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """RGB -> CIE Lab 색공간 변환 (NumPy).

    ΔE 계산에 필요하다.

    Args:
        image: RGB 이미지 [H, W, 3], float32, 범위 [0, 1]

    Returns:
        Lab 이미지 [H, W, 3]
    """
    return _rgb_to_lab_impl(np.asarray(image, dtype=np.float32))
