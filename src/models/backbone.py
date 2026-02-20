"""공용 backbone 네트워크.

Phase 2의 Image-Adaptive 3D LUT에서 사용할 경량 CNN.
입력 이미지를 분석하여 basis LUT 블렌딩 가중치를 예측한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNReLU(nn.Sequential):
    """Conv2d + BatchNorm2d + ReLU 블록."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class LightweightCNN(nn.Module):
    """이미지 분석용 경량 CNN backbone.

    저해상도 이미지(256x256)를 받아 basis LUT 블렌딩 가중치를 출력한다.
    4단계 stride=2 다운샘플 후 Global Average Pooling으로 공간 정보를 집약한다.

    파라미터 수 < 500K를 목표로 채널 수를 조절한다 (약 220K).

    Args:
        output_dim: 출력 차원 (basis LUT 개수와 동일)
    """

    def __init__(self, output_dim: int = 3) -> None:
        super().__init__()

        # 인코더: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        # 채널: 3 -> 16 -> 32 -> 64 -> 128 -> 128
        self.encoder = nn.Sequential(
            _ConvBNReLU(3,   16,  stride=2),   # 256 -> 128
            _ConvBNReLU(16,  16),
            _ConvBNReLU(16,  32,  stride=2),   # 128 -> 64
            _ConvBNReLU(32,  32),
            _ConvBNReLU(32,  64,  stride=2),   # 64  -> 32
            _ConvBNReLU(64,  64),
            _ConvBNReLU(64,  128, stride=2),   # 32  -> 16
            _ConvBNReLU(128, 128),
        )

        # Global Average Pooling + FC
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> [B, 128, 1, 1]
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """이미지에서 softmax 가중치 벡터 추출.

        Args:
            x: 입력 이미지 [B, 3, H, W], 범위 [0, 1]
               H, W는 256 권장이나 임의 크기 지원

        Returns:
            softmax 가중치 [B, output_dim], 합 = 1.0
        """
        feat = self.encoder(x)           # [B, 128, H/16, W/16]
        feat = self.gap(feat)            # [B, 128, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, 128]
        logits = self.fc(feat)           # [B, output_dim]
        return F.softmax(logits, dim=-1) # [B, output_dim], 합 = 1.0
