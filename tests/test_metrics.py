"""평가 지표 테스트."""

import pytest
import numpy as np


class TestMetrics:
    """평가 지표 계산 테스트."""

    def test_delta_e_identical(self):
        """동일 이미지의 ΔE == 0 검증."""
        # TODO: 동일한 이미지 2개로 compute_delta_e 호출
        # TODO: 결과 == 0.0 검증
        pass

    def test_psnr_identical(self):
        """동일 이미지의 PSNR == inf 검증."""
        # TODO: 동일한 이미지 2개로 compute_psnr 호출
        # TODO: 결과 == inf 검증
        pass

    def test_ssim_identical(self):
        """동일 이미지의 SSIM == 1.0 검증."""
        # TODO: 동일한 이미지 2개로 compute_ssim 호출
        # TODO: 결과 == 1.0 검증
        pass

    def test_delta_e_range(self):
        """ΔE가 비음수인지 검증."""
        # TODO: 랜덤 이미지 2개로 ΔE 계산
        # TODO: 결과 >= 0 검증
        pass
