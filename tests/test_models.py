"""모델 forward pass 테스트."""

import pytest
import torch


class TestNILUT:
    """NILUT 모델 테스트."""

    def test_single_style_forward(self):
        """단일 스타일 모델의 forward pass shape 검증."""
        # TODO: NILUT(num_styles=None) 생성
        # TODO: 랜덤 RGB 입력 [B, 3]으로 forward
        # TODO: 출력 shape == [B, 3] 검증
        # TODO: 출력 범위 [0, 1] 검증
        pass

    def test_multi_style_forward(self):
        """다중 스타일 모델의 forward pass shape 검증."""
        # TODO: NILUT(num_styles=10) 생성
        # TODO: style_idx 지정하여 forward
        # TODO: 출력 shape 검증
        pass

    def test_image_input_forward(self):
        """이미지 형태 입력 [B, H, W, 3] 처리 검증."""
        # TODO: [1, 64, 64, 3] 입력으로 forward
        # TODO: 출력 shape == [1, 64, 64, 3] 검증
        pass

    def test_style_blending(self):
        """스타일 블렌딩 검증."""
        # TODO: blend_styles 호출
        # TODO: 가중치 합 = 1.0 시 결과가 valid 범위인지 검증
        pass
