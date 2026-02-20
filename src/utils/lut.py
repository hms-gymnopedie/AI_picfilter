"""3D LUT 연산 유틸리티.

Trilinear interpolation 및 LUT 적용 함수.
NumPy(추론/평가)와 PyTorch(미분 가능 학습) 양쪽을 지원한다.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union


def apply_lut(
    image: np.ndarray,
    lut: np.ndarray,
) -> np.ndarray:
    """3D LUT를 이미지에 적용 (trilinear interpolation, NumPy).

    Args:
        image: 입력 이미지 [H, W, 3], float32, 범위 [0, 1]
        lut: 3D LUT [size, size, size, 3], float32

    Returns:
        변환된 이미지 [H, W, 3], float32
    """
    image = np.asarray(image, dtype=np.float32)
    lut = np.asarray(lut, dtype=np.float32)

    h, w, _ = image.shape
    size = lut.shape[0]
    scale = size - 1  # LUT 인덱스 스케일

    # 픽셀을 LUT 그리드 좌표로 변환
    coords = image * scale  # [H, W, 3], float
    lo = np.floor(coords).astype(np.int32).clip(0, size - 2)  # 하한 인덱스
    hi = lo + 1  # 상한 인덱스

    # 보간 가중치
    d = coords - lo  # [H, W, 3], 소수부
    d_r, d_g, d_b = d[..., 0], d[..., 1], d[..., 2]
    lo_r, lo_g, lo_b = lo[..., 0], lo[..., 1], lo[..., 2]
    hi_r, hi_g, hi_b = hi[..., 0], hi[..., 1], hi[..., 2]

    # 8개 꼭짓점의 LUT 값 조회 (R, G, B 순서)
    c000 = lut[lo_r, lo_g, lo_b]
    c001 = lut[lo_r, lo_g, hi_b]
    c010 = lut[lo_r, hi_g, lo_b]
    c011 = lut[lo_r, hi_g, hi_b]
    c100 = lut[hi_r, lo_g, lo_b]
    c101 = lut[hi_r, lo_g, hi_b]
    c110 = lut[hi_r, hi_g, lo_b]
    c111 = lut[hi_r, hi_g, hi_b]

    # Trilinear interpolation
    d_r = d_r[..., np.newaxis]
    d_g = d_g[..., np.newaxis]
    d_b = d_b[..., np.newaxis]

    c00 = c000 * (1 - d_b) + c001 * d_b
    c01 = c010 * (1 - d_b) + c011 * d_b
    c10 = c100 * (1 - d_b) + c101 * d_b
    c11 = c110 * (1 - d_b) + c111 * d_b

    c0 = c00 * (1 - d_g) + c01 * d_g
    c1 = c10 * (1 - d_g) + c11 * d_g

    result = c0 * (1 - d_r) + c1 * d_r
    return result.clip(0.0, 1.0)


def trilinear_interpolation(
    lut: torch.Tensor,
    rgb: torch.Tensor,
) -> torch.Tensor:
    """PyTorch trilinear interpolation (미분 가능).

    F.grid_sample을 사용하여 LUT lookup을 GPU에서 수행한다.

    Args:
        lut: 3D LUT [1, 3, size, size, size] (channel-first)
        rgb: 입력 RGB [B, 3], 범위 [0, 1]

    Returns:
        보간된 RGB [B, 3]
    """
    # F.grid_sample의 grid 좌표는 [-1, 1] 범위
    # rgb [0, 1] -> [-1, 1]
    grid = rgb * 2.0 - 1.0  # [B, 3]

    # grid_sample은 [B, 1, 1, 1, 3] 형태의 그리드를 기대
    # 좌표 순서: (x=R, y=G, z=B) -- grid_sample은 (x, y, z) 순서
    grid = grid.view(-1, 1, 1, 1, 3)  # [B, 1, 1, 1, 3]

    # lut: [1, 3, size, size, size]
    lut_batch = lut.expand(grid.shape[0], -1, -1, -1, -1)  # [B, 3, size, size, size]

    # grid_sample: input [B, C, D, H, W], grid [B, D_out, H_out, W_out, 3]
    out = F.grid_sample(
        lut_batch,
        grid,
        mode="bilinear",  # trilinear는 5D 입력에서 bilinear 모드로 동작
        padding_mode="border",
        align_corners=True,
    )  # [B, 3, 1, 1, 1]

    return out.view(-1, 3)  # [B, 3]
