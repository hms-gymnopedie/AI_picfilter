"""평가 지표 계산.

CIE76 ΔE, PSNR, SSIM 등 이미지 품질 평가 지표.
"""

import numpy as np
from typing import Optional

from src.utils.color import rgb_to_lab, delta_e_cie76


def compute_delta_e(image1: np.ndarray, image2: np.ndarray) -> float:
    """두 이미지 간 CIE76 ΔE 평균 계산.

    Args:
        image1: 첫 번째 이미지 [H, W, 3], float32, RGB, 범위 [0, 1]
        image2: 두 번째 이미지 [H, W, 3], float32, RGB, 범위 [0, 1]

    Returns:
        평균 ΔE 값. < 1.0이면 인지적으로 동일한 수준
    """
    image1 = np.asarray(image1, dtype=np.float32)
    image2 = np.asarray(image2, dtype=np.float32)

    lab1 = rgb_to_lab(image1)
    lab2 = rgb_to_lab(image2)

    delta = delta_e_cie76(lab1, lab2)  # [H, W]
    return float(delta.mean())


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """PSNR (Peak Signal-to-Noise Ratio) 계산.

    Args:
        original: 원본 이미지 [H, W, 3], float32, 범위 [0, 1]
        processed: 처리된 이미지 [H, W, 3], float32, 범위 [0, 1]

    Returns:
        PSNR 값 (dB). 높을수록 원본에 가까움. 40dB 이상이면 시각적으로 동일
    """
    original = np.asarray(original, dtype=np.float32)
    processed = np.asarray(processed, dtype=np.float32)

    mse = float(np.mean((original - processed) ** 2))

    if mse < 1e-10:
        return float("inf")

    # PSNR = 10 * log10(MAX^2 / MSE), MAX = 1.0
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(
    original: np.ndarray,
    processed: np.ndarray,
    window_size: int = 11,
) -> float:
    """SSIM (Structural Similarity Index) 계산.

    채널별 SSIM을 계산한 뒤 평균을 반환한다.
    Gaussian 윈도우를 사용한다.

    Args:
        original: 원본 이미지 [H, W, 3], float32, 범위 [0, 1]
        processed: 처리된 이미지 [H, W, 3], float32, 범위 [0, 1]
        window_size: SSIM Gaussian 윈도우 크기 (홀수 권장)

    Returns:
        SSIM 값 [0, 1]. 1에 가까울수록 구조가 보존됨
    """
    from scipy.ndimage import uniform_filter

    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)

    # SSIM 상수 (L=1.0 기준)
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    def _ssim_channel(x: np.ndarray, y: np.ndarray) -> float:
        """단일 채널 SSIM 계산."""
        size = window_size

        mu_x = uniform_filter(x, size=size)
        mu_y = uniform_filter(y, size=size)

        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x2 = uniform_filter(x ** 2, size=size) - mu_x2
        sigma_y2 = uniform_filter(y ** 2, size=size) - mu_y2
        sigma_xy = uniform_filter(x * y, size=size) - mu_xy

        # SSIM 공식
        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_map = num / (den + 1e-10)

        return float(ssim_map.mean())

    ssim_values = [
        _ssim_channel(original[..., c], processed[..., c])
        for c in range(original.shape[-1])
    ]

    return float(np.mean(ssim_values))
