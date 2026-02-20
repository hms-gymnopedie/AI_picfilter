"""데이터 변환 테스트."""

import pytest
import numpy as np
import torch


class TestColorTransforms:
    """색공간 변환 테스트."""

    def test_srgb_linear_roundtrip(self):
        """sRGB -> Linear -> sRGB 왕복 검증."""
        # TODO: srgb_to_linear -> linear_to_srgb 왕복
        # TODO: 원본과 일치 (torch.allclose)
        pass

    def test_rgb_to_lab_white(self):
        """백색(1,1,1)의 Lab 변환 검증."""
        # TODO: [1.0, 1.0, 1.0] -> Lab
        # TODO: L ~= 100, a ~= 0, b ~= 0 검증
        pass

    def test_rgb_to_lab_black(self):
        """흑색(0,0,0)의 Lab 변환 검증."""
        # TODO: [0.0, 0.0, 0.0] -> Lab
        # TODO: L ~= 0, a ~= 0, b ~= 0 검증
        pass
