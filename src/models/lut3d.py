"""Image-Adaptive 3D LUT 모델.

경량 CNN이 이미지를 분석해 N개의 basis LUT 블렌딩 가중치를 예측하고,
가중 합산된 LUT를 trilinear interpolation으로 원본 해상도에 적용한다.

Reference: HuiZeng/Image-Adaptive-3DLUT (Apache 2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.backbone import LightweightCNN


class ImageAdaptive3DLUT(nn.Module):
    """Image-Adaptive 3D LUT 모델.

    구조:
        backbone(저해상도 이미지) -> softmax weights [B, n_basis]
        weights @ basis_luts           -> blended_lut [lut_size^3, 3]
        trilinear interpolation        -> 변환된 이미지 [B, 3, H, W]

    basis_luts는 학습 가능한 파라미터로, 항등 LUT에서 초기화된다.
    backbone은 이미지 내용에 따라 각 basis의 기여도를 동적으로 결정한다.

    Args:
        n_basis: basis LUT 개수 (보통 3~5)
        lut_size: 3D LUT 그리드 크기 (17 또는 33)
        backbone_input_size: backbone에 입력할 이미지 크기 (다운샘플 후)
    """

    def __init__(
        self,
        n_basis: int = 3,
        lut_size: int = 33,
        backbone_input_size: int = 256,
    ) -> None:
        super().__init__()

        self.n_basis = n_basis
        self.lut_size = lut_size
        self.backbone_input_size = backbone_input_size

        # 학습 가능한 basis LUTs: [n_basis, lut_size, lut_size, lut_size, 3]
        # 항등 변환으로 초기화 (출력 = 입력)
        identity = self._make_identity_lut(lut_size)  # [lut_size^3, 3]
        basis = identity.unsqueeze(0).repeat(n_basis, 1, 1)  # [n_basis, lut_size^3, 3]
        self.basis_luts = nn.Parameter(basis)

        # backbone: 이미지 -> [B, n_basis] softmax 가중치
        self.backbone = LightweightCNN(output_dim=n_basis)

    @staticmethod
    def _make_identity_lut(lut_size: int) -> torch.Tensor:
        """항등 LUT 생성 [lut_size^3, 3]."""
        coords = torch.linspace(0.0, 1.0, lut_size)
        r, g, b = torch.meshgrid(coords, coords, coords, indexing="ij")
        return torch.stack([r.ravel(), g.ravel(), b.ravel()], dim=-1)  # [N, 3]

    def _apply_lut_to_image(
        self,
        image: torch.Tensor,
        lut_flat: torch.Tensor,
    ) -> torch.Tensor:
        """blended LUT를 이미지에 trilinear interpolation으로 적용.

        Args:
            image: [B, 3, H, W], 범위 [0, 1]
            lut_flat: [lut_size^3, 3] — 단일 LUT (픽셀 독립 변환)

        Returns:
            변환된 이미지 [B, 3, H, W]
        """
        b, c, h, w = image.shape
        size = self.lut_size

        # LUT를 [1, 3, D, H, W] channel-first 5D 텐서로 변환
        # lut_flat: [size^3, 3] -> [size, size, size, 3] -> [1, 3, size, size, size]
        lut_5d = lut_flat.reshape(size, size, size, 3)
        lut_5d = lut_5d.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, S, S, S]

        # 이미지 픽셀 좌표를 [-1, 1] 범위로 변환
        # image: [B, 3, H, W] -> 픽셀 좌표 [B, H*W, 3]
        pixels = image.permute(0, 2, 3, 1).reshape(b, h * w, 3)  # [B, N, 3]
        grid = pixels * 2.0 - 1.0                                  # [-1, 1]

        # grid_sample: [B, 1, 1, N, 3] 형태로 N개 포인트 동시 조회
        grid_5d = grid.view(b, 1, 1, h * w, 3)   # [B, 1, 1, N, 3]
        lut_expanded = lut_5d.expand(b, -1, -1, -1, -1)  # [B, 3, S, S, S]

        sampled = F.grid_sample(
            lut_expanded,
            grid_5d,
            mode="bilinear",        # 5D에서 trilinear로 동작
            padding_mode="border",
            align_corners=True,
        )  # [B, 3, 1, 1, N]

        out = sampled.view(b, 3, h, w)
        return out.clamp(0.0, 1.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """이미지 적응형 LUT 변환 수행.

        Args:
            image: 입력 이미지 [B, 3, H, W], 범위 [0, 1]
                   H, W는 임의 크기 가능 (4K 포함)

        Returns:
            변환된 이미지 [B, 3, H, W]
        """
        b = image.shape[0]

        # backbone 입력: 저해상도로 다운샘플
        size = self.backbone_input_size
        if image.shape[-1] != size or image.shape[-2] != size:
            thumb = F.interpolate(
                image, size=(size, size), mode="bilinear", align_corners=False
            )
        else:
            thumb = image

        # backbone -> softmax weights [B, n_basis]
        weights = self.backbone(thumb)  # [B, n_basis]

        # basis LUT 가중 합산: [B, n_basis] x [n_basis, lut_size^3, 3] -> [B, lut_size^3, 3]
        # basis_luts: [n_basis, lut_size^3, 3]
        # weights: [B, n_basis] -> [B, n_basis, 1, 1] 브로드캐스트
        blended = torch.einsum("bn,npc->bpc", weights, self.basis_luts)  # [B, lut_size^3, 3]

        # 각 이미지별로 blended LUT 적용
        # 배치 내 이미지마다 LUT가 다르므로 개별 처리
        results = []
        for i in range(b):
            out_i = self._apply_lut_to_image(
                image[i:i+1],   # [1, 3, H, W]
                blended[i],     # [lut_size^3, 3]
            )
            results.append(out_i)

        return torch.cat(results, dim=0)  # [B, 3, H, W]

    def get_blended_lut(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """이미지에 대한 blended LUT 반환 (.cube export용).

        Args:
            image: 입력 이미지 [1, 3, H, W]

        Returns:
            blended LUT [lut_size, lut_size, lut_size, 3]
        """
        size = self.backbone_input_size
        if image.shape[-1] != size or image.shape[-2] != size:
            thumb = F.interpolate(
                image, size=(size, size), mode="bilinear", align_corners=False
            )
        else:
            thumb = image

        with torch.no_grad():
            weights = self.backbone(thumb)  # [1, n_basis]
            blended = torch.einsum("bn,npc->bpc", weights, self.basis_luts)[0]  # [lut_size^3, 3]

        return blended.reshape(self.lut_size, self.lut_size, self.lut_size, 3)
