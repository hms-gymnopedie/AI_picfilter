"""손실 함수 모음.

NILUT 및 Image-Adaptive 3D LUT 모델 학습에 필요한 손실 함수들.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.color import rgb_to_lab as _rgb_to_lab


class DeltaELoss(nn.Module):
    """CIE76 ΔE 기반 손실 함수.

    Lab 색공간에서의 유클리드 거리를 손실로 사용한다.
    ΔE < 1.0이면 인간이 색상 차이를 인지하지 못하는 수준이다.
    PyTorch 자동미분을 통해 역전파가 가능하다.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        """ΔE 손실 계산.

        Args:
            pred_rgb: 예측 RGB [B, 3] 또는 [..., 3], 범위 [0, 1]
            target_rgb: 목표 RGB, 동일 shape

        Returns:
            배치 평균 ΔE 값 (scalar)
        """
        pred_lab = _rgb_to_lab(pred_rgb.clamp(0.0, 1.0))
        target_lab = _rgb_to_lab(target_rgb.clamp(0.0, 1.0))

        # CIE76: 유클리드 거리
        delta_e = torch.sqrt(((pred_lab - target_lab) ** 2).sum(dim=-1) + 1e-8)
        return delta_e.mean()


class SmoothnessLoss(nn.Module):
    """LUT 평활도(smoothness) 정규화 손실.

    학습된 색상 변환이 급격한 불연속 없이 부드럽게 변하도록 제약한다.
    균일한 RGB 그리드를 생성하여 인접 색상 간 변환 차이를 최소화한다.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, model: nn.Module, grid_size: int = 17) -> torch.Tensor:
        """평활도 손실 계산.

        Args:
            model: NILUT 모델 (forward(rgb) 호출 가능)
            grid_size: 평가용 그리드 크기 (커질수록 정밀하지만 느림)

        Returns:
            평활도 손실 값 (scalar)
        """
        # 균일 RGB 그리드 생성 [grid_size^3, 3]
        coords = torch.linspace(0.0, 1.0, grid_size)
        r, g, b = torch.meshgrid(coords, coords, coords, indexing="ij")
        grid = torch.stack([r.ravel(), g.ravel(), b.ravel()], dim=-1)  # [N, 3]

        # 모델 파라미터와 동일 device로 이동
        device = next(model.parameters()).device
        grid = grid.to(device)

        with torch.enable_grad():
            # 다중 스타일 모드에서는 스타일 0을 기준으로 평활도 계산
            num_styles = getattr(model, "num_styles", None)
            if num_styles is not None:
                style_tensor = torch.zeros(grid.shape[0], dtype=torch.long, device=device)
                out = model(grid, style_idx=style_tensor)  # [N, 3]
            else:
                out = model(grid)  # [N, 3]

        # [grid_size, grid_size, grid_size, 3] 로 reshape
        out_3d = out.reshape(grid_size, grid_size, grid_size, 3)

        # 인접 점 간 차이 (R, G, B 각 방향)
        diff_r = (out_3d[1:, :, :, :] - out_3d[:-1, :, :, :]) ** 2  # R 방향
        diff_g = (out_3d[:, 1:, :, :] - out_3d[:, :-1, :, :]) ** 2  # G 방향
        diff_b = (out_3d[:, :, 1:, :] - out_3d[:, :, :-1, :]) ** 2  # B 방향

        loss = diff_r.mean() + diff_g.mean() + diff_b.mean()
        return loss


class PerceptualLoss(nn.Module):
    """VGG16 기반 지각적 손실 함수 (Perceptual Loss).

    relu2_2와 relu3_3 레이어의 feature map L1 거리를 손실로 사용한다.
    색상 변환이 이미지 구조(텍스처, 엣지)를 보존하도록 제약한다.

    Note:
        torchvision이 설치되어 있어야 한다.
        VGG16 가중치는 자동으로 다운로드된다.
    """

    def __init__(self, layers: list[str] | None = None) -> None:
        """
        Args:
            layers: 사용할 VGG16 레이어 이름 목록.
                    None이면 ['relu2_2', 'relu3_3'] 사용.
        """
        super().__init__()
        self._layers = layers or ["relu2_2", "relu3_3"]
        self._vgg: nn.Module | None = None  # lazy 로드

    def _build_vgg(self, device: torch.device) -> None:
        """VGG16 feature extractor를 lazy하게 초기화."""
        try:
            from torchvision import models
        except ImportError:
            raise ImportError("torchvision이 필요합니다: pip install torchvision")

        # VGG16 레이어 이름 -> 인덱스 매핑
        # relu2_2: features[9], relu3_3: features[16]
        _layer_map = {"relu2_2": 9, "relu3_3": 16, "relu4_3": 23}

        max_idx = max(_layer_map[layer] for layer in self._layers) + 1
        vgg_full = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self._vgg = nn.Sequential(*list(vgg_full.features.children())[:max_idx])
        self._vgg.to(device)
        for p in self._vgg.parameters():
            p.requires_grad_(False)
        self._vgg.eval()

        # 사용할 feature 인덱스 저장
        self._feature_indices = [_layer_map[layer] for layer in self._layers]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """지각적 손실 계산.

        Args:
            pred: 예측 이미지 [B, 3, H, W], 범위 [0, 1]
            target: 목표 이미지 [B, 3, H, W], 범위 [0, 1]

        Returns:
            지각적 손실 (scalar)
        """
        device = pred.device

        if self._vgg is None:
            self._build_vgg(device)
        else:
            self._vgg = self._vgg.to(device)

        # ImageNet 정규화
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        pred_norm = (pred.clamp(0.0, 1.0) - mean) / std
        target_norm = (target.clamp(0.0, 1.0) - mean) / std

        loss = torch.tensor(0.0, device=device)
        feat_idx_set = set(self._feature_indices)

        x_pred = pred_norm
        x_tgt = target_norm

        for i, layer in enumerate(self._vgg):
            x_pred = layer(x_pred)
            x_tgt = layer(x_tgt)
            if i in feat_idx_set:
                loss = loss + F.l1_loss(x_pred, x_tgt.detach())

        return loss


class MonotonicityLoss(nn.Module):
    """LUT 단조성(Monotonicity) 정규화 손실.

    각 채널에서 입력 값이 증가하면 출력 값도 증가해야 한다는 제약이다.
    인접 LUT 셀 간 출력 차이가 음수일 때 페널티를 부과한다.

    자연스러운 색상 변환은 단조성을 가지므로, 이 제약을 위반하면 색상 역전
    아티팩트가 발생한다.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, basis_luts: torch.Tensor) -> torch.Tensor:
        """단조성 손실 계산.

        Args:
            basis_luts: basis LUT 파라미터 [n_basis, lut_size^3, 3]
                        또는 [n_basis, lut_size, lut_size, lut_size, 3]

        Returns:
            단조성 위반 페널티 (scalar)
        """
        # [n_basis, lut_size^3, 3] -> [n_basis, lut_size, lut_size, lut_size, 3]
        if basis_luts.ndim == 3:
            n, flat, c = basis_luts.shape
            size = round(flat ** (1 / 3))
            lut = basis_luts.reshape(n, size, size, size, c)
        else:
            lut = basis_luts
            size = lut.shape[1]

        # R, G, B 각 방향에서 인접 셀 간 차이 계산
        # 단조 증가: diff >= 0이어야 함 → 음수 차이에만 페널티
        diff_r = lut[:, 1:, :, :, :] - lut[:, :-1, :, :, :]  # R 방향
        diff_g = lut[:, :, 1:, :, :] - lut[:, :, :-1, :, :]  # G 방향
        diff_b = lut[:, :, :, 1:, :] - lut[:, :, :, :-1, :]  # B 방향

        # 음수 차이의 제곱합 (ReLU(-x) = max(0, -x))
        penalty_r = F.relu(-diff_r).pow(2).mean()
        penalty_g = F.relu(-diff_g).pow(2).mean()
        penalty_b = F.relu(-diff_b).pow(2).mean()

        return penalty_r + penalty_g + penalty_b


class LUT3DCombinedLoss(nn.Module):
    """Image-Adaptive 3D LUT 학습용 복합 손실.

    L1 재구성 손실 + 지각적 손실 + 단조성 정규화를 결합한다.
    paired(보정 전/후 이미지 쌍) 학습에 적합하다.

    Args:
        lambda_perceptual: 지각적 손실 가중치
        lambda_monotonic: 단조성 손실 가중치
        use_perceptual: VGG 지각적 손실 사용 여부 (torchvision 필요)
    """

    def __init__(
        self,
        lambda_perceptual: float = 0.1,
        lambda_monotonic: float = 10.0,
        use_perceptual: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_perceptual = lambda_perceptual
        self.lambda_monotonic = lambda_monotonic
        self.use_perceptual = use_perceptual

        self.l1_loss = nn.L1Loss()
        self.mono_loss = MonotonicityLoss()
        self.perceptual_loss = PerceptualLoss() if use_perceptual else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        basis_luts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """복합 손실 계산.

        Args:
            pred: 예측 이미지 [B, 3, H, W]
            target: 목표 이미지 [B, 3, H, W]
            basis_luts: basis LUT 파라미터 [n_basis, lut_size^3, 3]

        Returns:
            dict: {'total': ..., 'l1': ..., 'perceptual': ..., 'monotonic': ...}
        """
        l1 = self.l1_loss(pred, target)
        mono = self.mono_loss(basis_luts)

        loss_dict = {
            "l1": l1.detach(),
            "monotonic": mono.detach(),
        }

        total = l1 + self.lambda_monotonic * mono

        if self.use_perceptual and self.perceptual_loss is not None:
            perc = self.perceptual_loss(pred, target)
            total = total + self.lambda_perceptual * perc
            loss_dict["perceptual"] = perc.detach()
        else:
            loss_dict["perceptual"] = torch.tensor(0.0, device=pred.device)

        loss_dict["total"] = total
        return loss_dict


class CombinedLoss(nn.Module):
    """복합 손실 함수.

    L1 재구성 손실 + ΔE 색상 손실 + 평활도 정규화를 결합한다.

    Args:
        lambda_delta_e: ΔE 손실 가중치
        lambda_smooth: 평활도 손실 가중치
        smooth_grid_size: 평활도 계산용 그리드 크기
    """

    def __init__(
        self,
        lambda_delta_e: float = 0.5,
        lambda_smooth: float = 0.01,
        smooth_grid_size: int = 17,
    ) -> None:
        super().__init__()
        self.lambda_delta_e = lambda_delta_e
        self.lambda_smooth = lambda_smooth
        self.smooth_grid_size = smooth_grid_size

        self.l1_loss = nn.L1Loss()
        self.delta_e_loss = DeltaELoss()
        self.smooth_loss = SmoothnessLoss()

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        model: nn.Module,
    ) -> dict[str, torch.Tensor]:
        """복합 손실 계산.

        Args:
            pred_rgb: 예측 RGB
            target_rgb: 목표 RGB
            model: NILUT 모델 (평활도 계산용)

        Returns:
            dict: {'total': 합산 손실, 'l1': ..., 'delta_e': ..., 'smooth': ...}
        """
        l1 = self.l1_loss(pred_rgb, target_rgb)
        de = self.delta_e_loss(pred_rgb, target_rgb)
        smooth = self.smooth_loss(model, self.smooth_grid_size)

        total = l1 + self.lambda_delta_e * de + self.lambda_smooth * smooth

        return {
            "total": total,
            "l1": l1.detach(),
            "delta_e": de.detach(),
            "smooth": smooth.detach(),
        }
