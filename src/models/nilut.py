"""NILUT (Neural Implicit LUT) 모델.

MLP 기반의 연속 색상 변환 모델.
입력 RGB 좌표를 받아 변환된 RGB를 출력한다.
단일 모델로 다중 스타일을 지원하며, 스타일 간 연속적 블렌딩이 가능하다.

Reference: mv-lab/nilut (MIT License)
"""

import torch
import torch.nn as nn
from typing import Optional


class NILUT(nn.Module):
    """Neural Implicit LUT 모델.

    구조:
        [RGB (3) + style_emb (style_dim, optional)] -> MLP -> [RGB (3)]

    단일 스타일 모드: style_idx / style_weights 불필요
    다중 스타일 모드: num_styles 지정, style_idx 또는 style_weights로 스타일 선택/블렌딩

    Args:
        hidden_dims: 은닉층 차원 리스트. 예: [256, 256, 256]
        activation: 활성화 함수. 'relu' 또는 'gelu'
        num_styles: 지원할 스타일 수. None이면 단일 스타일 모드
        style_dim: 스타일 임베딩 차원 (다중 스타일 모드에서만 사용)
    """

    def __init__(
        self,
        hidden_dims: list[int] = [256, 256, 256],
        activation: str = "gelu",
        num_styles: Optional[int] = None,
        style_dim: int = 64,
    ) -> None:
        super().__init__()

        self.num_styles = num_styles
        self.style_dim = style_dim if num_styles is not None else 0

        # 스타일 임베딩 (다중 스타일 모드)
        self.style_embedding: Optional[nn.Embedding] = None
        if num_styles is not None:
            self.style_embedding = nn.Embedding(num_styles, style_dim)
            nn.init.normal_(self.style_embedding.weight, std=0.01)

        # 입력 차원: RGB(3) + style_emb(style_dim)
        in_dim = 3 + self.style_dim

        # 활성화 함수 선택
        act_map = {"relu": nn.ReLU, "gelu": nn.GELU}
        if activation not in act_map:
            raise ValueError(
                f"지원하지 않는 활성화 함수: {activation}. 'relu' 또는 'gelu'만 허용"
            )
        Act = act_map[activation]

        # MLP 구성: Linear -> Act -> ... -> Linear -> Sigmoid
        layers: list[nn.Module] = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(Act())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 3))
        layers.append(nn.Sigmoid())  # 출력 범위를 [0, 1]로 고정

        self.mlp = nn.Sequential(*layers)

        # 가중치 초기화 (He init은 ReLU 계열에 적합)
        self._init_weights()

    def _init_weights(self) -> None:
        """MLP 가중치 초기화."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        rgb: torch.Tensor,
        style_idx: Optional[torch.Tensor] = None,
        style_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """색상 변환 수행.

        Args:
            rgb: 입력 RGB 텐서 [B, 3] 또는 [B, H, W, 3] 또는 [B, 3, H, W], 범위 [0, 1]
            style_idx: 스타일 인덱스 [B] (다중 스타일 모드, 정수 인덱스)
            style_weights: 스타일 블렌딩 가중치 [B, num_styles] (블렌딩 모드)
                           style_idx와 style_weights는 동시에 지정하지 않는다.

        Returns:
            변환된 RGB 텐서, 입력과 동일한 shape
        """
        # 입력 shape 기록 및 [N, 3] 픽셀 배치로 변환
        is_image_chw = False

        if rgb.ndim == 4 and rgb.shape[1] == 3:
            # [B, 3, H, W] -> [B, H, W, 3] -> [B*H*W, 3]
            is_image_chw = True
            b, c, h, w = rgb.shape
            rgb_flat = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
        elif rgb.ndim == 4:
            # [B, H, W, 3]
            b, h, w, c = rgb.shape
            rgb_flat = rgb.reshape(-1, 3)
        elif rgb.ndim == 2:
            # [B, 3]
            rgb_flat = rgb
            b = rgb.shape[0]
        else:
            raise ValueError(f"지원하지 않는 RGB shape: {rgb.shape}")

        n_pixels = rgb_flat.shape[0]

        # 스타일 조건 결합
        if self.style_embedding is not None:
            if style_weights is not None:
                # 블렌딩 모드: 가중치 합산으로 스타일 임베딩 계산
                # style_weights: [B, num_styles]
                all_emb = self.style_embedding.weight  # [num_styles, style_dim]
                # [B, num_styles] x [num_styles, style_dim] -> [B, style_dim]
                emb = style_weights @ all_emb
            elif style_idx is not None:
                emb = self.style_embedding(style_idx)  # [B, style_dim]
            else:
                raise ValueError(
                    "다중 스타일 모드에서는 style_idx 또는 style_weights가 필요합니다."
                )

            # 픽셀 수에 맞게 임베딩 반복 확장
            # emb: [B, style_dim] -> [n_pixels, style_dim]
            if rgb.ndim == 4:
                pixels_per_batch = n_pixels // b
                emb = (
                    emb.unsqueeze(1)
                    .expand(-1, pixels_per_batch, -1)
                    .reshape(-1, self.style_dim)
                )
            # [B, 3]인 경우는 B == n_pixels이므로 그대로 사용

            x = torch.cat([rgb_flat, emb], dim=-1)
        else:
            x = rgb_flat

        # MLP forward pass
        out_flat = self.mlp(x)  # [n_pixels, 3]

        # 출력 shape 복원
        if rgb.ndim == 4 and is_image_chw:
            out = out_flat.reshape(b, h, w, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
        elif rgb.ndim == 4:
            out = out_flat.reshape(b, h, w, 3)
        else:
            out = out_flat  # [B, 3]

        return out

    def blend_styles(
        self,
        rgb: torch.Tensor,
        style_indices: list[int],
        weights: list[float],
    ) -> torch.Tensor:
        """여러 스타일을 지정된 가중치로 블렌딩하여 적용.

        스타일 임베딩 공간에서 선형 보간한 뒤 단일 forward pass를 수행한다.
        이는 각 스타일을 독립적으로 적용한 뒤 출력을 평균하는 것과 다르다.
        임베딩 공간의 보간이 더 자연스러운 블렌딩을 제공한다.

        Args:
            rgb: 입력 RGB 텐서 [B, 3] 또는 이미지 shape
            style_indices: 블렌딩할 스타일 인덱스 리스트
            weights: 각 스타일의 가중치 (합 = 1.0이 권장)

        Returns:
            블렌딩된 스타일이 적용된 RGB 텐서
        """
        if self.style_embedding is None:
            raise RuntimeError("blend_styles는 다중 스타일 모드에서만 사용 가능합니다.")

        if len(style_indices) != len(weights):
            raise ValueError("style_indices와 weights의 길이가 같아야 합니다.")

        # 정규화
        total = sum(weights)
        norm_weights = [w / total for w in weights]

        # 가중치 텐서 구성 [1, num_styles] (0으로 초기화)
        weight_tensor = torch.zeros(
            1, self.num_styles, device=rgb.device, dtype=rgb.dtype
        )
        for idx, w in zip(style_indices, norm_weights):
            weight_tensor[0, idx] = w

        # 배치 크기에 맞게 확장
        b = rgb.shape[0] if rgb.ndim >= 2 else 1
        weight_tensor = weight_tensor.expand(b, -1)

        return self.forward(rgb, style_weights=weight_tensor)
