"""LUT 연산 테스트."""

import pytest
import numpy as np


class TestLUT:
    """3D LUT 연산 테스트."""

    def test_identity_lut_preserves_image(self):
        """항등 LUT 적용 시 이미지 불변 검증."""
        # TODO: 항등 LUT 생성
        # TODO: 랜덤 이미지에 적용
        # TODO: 입출력 일치 검증 (np.allclose, atol=1e-3)
        pass

    def test_trilinear_interpolation_corners(self):
        """LUT 코너 값의 정확한 룩업 검증."""
        # TODO: 항등 LUT의 코너 좌표(0, 0.5, 1) 조회
        # TODO: 입력과 출력이 일치하는지 검증
        pass
