"""색공간 변환 유틸리티.

sRGB, Linear RGB, CIE XYZ, CIE Lab 간 변환.
NumPy(CPU 평가) 및 PyTorch(미분 가능 학습 루프) 양쪽을 지원한다.
"""

import numpy as np
import torch
from typing import Union

# D65 조명 기준 백색점
D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

# sRGB -> XYZ (D65) 변환 행렬
_RGB_TO_XYZ_MAT = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)

# PyTorch 버전 (상수 등록)
_RGB_TO_XYZ_MAT_T = torch.tensor(_RGB_TO_XYZ_MAT.T, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------


def _srgb_to_linear_np(x: np.ndarray) -> np.ndarray:
    """sRGB 감마 디코딩 (numpy)."""
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb_to_linear_torch(x: torch.Tensor) -> torch.Tensor:
    """sRGB 감마 디코딩 (torch, 미분 가능)."""
    x = x.clamp(0.0, 1.0)
    linear = torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055).pow(2.4),
    )
    return linear


def _xyz_f(t: Union[np.ndarray, torch.Tensor], is_torch: bool = False):
    """CIE Lab의 비선형 f(t) 함수."""
    delta = 6.0 / 29.0
    delta3 = delta**3  # ≈ 0.008856
    c1 = 1.0 / (3.0 * delta**2)  # ≈ 7.787
    c2 = 4.0 / 29.0  # ≈ 0.1379
    if is_torch:
        return torch.where(t > delta3, t.clamp(min=1e-8).pow(1.0 / 3.0), c1 * t + c2)
    return np.where(t > delta3, np.cbrt(t.clip(1e-8)), c1 * t + c2)


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


def rgb_to_xyz(rgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """sRGB -> CIE XYZ 변환 (D65 기준).

    Args:
        rgb: sRGB 값, shape [..., 3], 범위 [0, 1]

    Returns:
        XYZ 값, 동일 shape
    """
    if isinstance(rgb, torch.Tensor):
        linear = _srgb_to_linear_torch(rgb)
        mat = _RGB_TO_XYZ_MAT_T.to(rgb.device)
        return linear @ mat  # [..., 3] x [3, 3] -> [..., 3]
    else:
        rgb = np.asarray(rgb, dtype=np.float32)
        linear = _srgb_to_linear_np(rgb)
        return linear @ _RGB_TO_XYZ_MAT.T


def xyz_to_lab(xyz: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """CIE XYZ -> CIE Lab 변환 (D65 기준).

    Args:
        xyz: XYZ 값, shape [..., 3]

    Returns:
        Lab 값: L [0, 100], a [-128, 127], b [-128, 127]
    """
    if isinstance(xyz, torch.Tensor):
        white = torch.tensor(D65_WHITE, dtype=xyz.dtype, device=xyz.device)
        t = xyz / white
        f = _xyz_f(t, is_torch=True)
        L = 116.0 * f[..., 1] - 16.0
        a = 500.0 * (f[..., 0] - f[..., 1])
        b = 200.0 * (f[..., 1] - f[..., 2])
        return torch.stack([L, a, b], dim=-1)
    else:
        xyz = np.asarray(xyz, dtype=np.float32)
        t = xyz / D65_WHITE
        f = _xyz_f(t, is_torch=False)
        L = 116.0 * f[..., 1] - 16.0
        a = 500.0 * (f[..., 0] - f[..., 1])
        b = 200.0 * (f[..., 1] - f[..., 2])
        return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """sRGB -> CIE Lab 변환 (편의 래퍼).

    Args:
        rgb: sRGB 값, shape [..., 3], 범위 [0, 1]

    Returns:
        Lab 값, 동일 shape
    """
    return xyz_to_lab(rgb_to_xyz(rgb))


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76 ΔE 계산.

    Args:
        lab1: 첫 번째 Lab 값, shape [..., 3]
        lab2: 두 번째 Lab 값, shape [..., 3]

    Returns:
        ΔE 값, shape [...] (마지막 축 제거)
    """
    diff = np.asarray(lab1, dtype=np.float32) - np.asarray(lab2, dtype=np.float32)
    return np.sqrt((diff**2).sum(axis=-1))
