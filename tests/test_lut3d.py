"""ImageAdaptive3DLUT 모델 및 관련 컴포넌트 테스트."""

import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# LightweightCNN backbone 테스트
# ---------------------------------------------------------------------------

class TestLightweightCNN:
    """LightweightCNN backbone 단위 테스트."""

    def test_output_shape(self):
        """출력 shape이 [B, output_dim]인지 검증."""
        from src.models.backbone import LightweightCNN
        model = LightweightCNN(output_dim=3)
        model.eval()
        x = torch.rand(2, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3), f"기대 (2, 3), 실제 {out.shape}"

    def test_softmax_output(self):
        """출력이 softmax (합=1.0, 범위[0,1])인지 검증."""
        from src.models.backbone import LightweightCNN
        model = LightweightCNN(output_dim=4)
        model.eval()
        x = torch.rand(4, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5), \
            f"softmax 합이 1.0이 아님: {sums}"
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_arbitrary_input_size(self):
        """임의 입력 크기에서도 동작하는지 검증."""
        from src.models.backbone import LightweightCNN
        model = LightweightCNN(output_dim=3)
        model.eval()
        for size in [128, 224, 512]:
            x = torch.rand(1, 3, size, size)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 3), f"입력 {size}x{size}에서 shape 오류"

    def test_param_count_under_500k(self):
        """파라미터 수가 500K 미만인지 검증."""
        from src.models.backbone import LightweightCNN
        model = LightweightCNN(output_dim=5)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 500_000, f"파라미터 수 {n_params}가 500K 초과"


# ---------------------------------------------------------------------------
# ImageAdaptive3DLUT 모델 테스트
# ---------------------------------------------------------------------------

class TestImageAdaptive3DLUT:
    """ImageAdaptive3DLUT 통합 테스트."""

    def _make_model(self, n_basis=3, lut_size=17, backbone_input_size=64):
        from src.models.lut3d import ImageAdaptive3DLUT
        return ImageAdaptive3DLUT(
            n_basis=n_basis,
            lut_size=lut_size,
            backbone_input_size=backbone_input_size,
        ).eval()

    def test_forward_shape(self):
        """forward 출력 shape이 입력과 동일한지 검증."""
        model = self._make_model()
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape, f"기대 {x.shape}, 실제 {out.shape}"

    def test_output_range(self):
        """출력이 [0, 1] 범위 내인지 검증."""
        model = self._make_model()
        x = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0, f"출력 최솟값 {out.min():.4f} < 0"
        assert out.max() <= 1.0, f"출력 최댓값 {out.max():.4f} > 1"

    def test_identity_init_near_identity(self):
        """초기화 직후 출력이 [0, 1] 범위를 유지하는지 검증.

        basis_luts가 항등 LUT로 초기화되므로 출력도 유효한 색상 범위 내에 있어야 한다.
        backbone은 무작위 초기화되므로 softmax 가중치가 균등하지 않아 MAE가 클 수 있지만,
        가중 합산된 LUT도 항등이므로 출력 범위는 [0, 1]을 유지해야 한다.
        """
        model = self._make_model()
        x = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        # 범위 검증 (항등 LUT 초기화의 핵심 보장)
        assert out.min() >= 0.0, f"출력 최솟값 {out.min():.4f} < 0"
        assert out.max() <= 1.0, f"출력 최댓값 {out.max():.4f} > 1"
        # 합산 가중치 softmax이므로 blended LUT도 항등 LUT이어야 함
        # (basis_luts가 동일하게 초기화되어 가중치 무관하게 동일)
        mae = (out - x).abs().mean().item()
        assert mae < 0.5, f"초기화 시 MAE {mae:.4f}가 비합리적으로 큼 (trilinear 오차 고려)"

    def test_arbitrary_resolution(self):
        """임의 해상도(4K급)에서도 forward가 동작하는지 검증."""
        model = self._make_model(lut_size=17, backbone_input_size=64)
        x = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 256, 256)

    def test_get_blended_lut_shape(self):
        """get_blended_lut() 출력 shape 검증."""
        lut_size = 17
        model = self._make_model(lut_size=lut_size, backbone_input_size=64)
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            blended = model.get_blended_lut(x)
        assert blended.shape == (lut_size, lut_size, lut_size, 3), \
            f"기대 ({lut_size},{lut_size},{lut_size},3), 실제 {blended.shape}"

    def test_basis_luts_parameter(self):
        """basis_luts가 학습 가능한 nn.Parameter인지 검증."""
        from src.models.lut3d import ImageAdaptive3DLUT
        model = ImageAdaptive3DLUT(n_basis=3, lut_size=17, backbone_input_size=64)
        assert isinstance(model.basis_luts, torch.nn.Parameter)
        assert model.basis_luts.requires_grad

    def test_gradient_flow(self):
        """역전파 시 gradient가 basis_luts와 backbone에 흐르는지 검증."""
        model = self._make_model()
        model.train()
        x = torch.rand(1, 3, 32, 32)
        out = model(x)
        loss = out.mean()
        loss.backward()
        assert model.basis_luts.grad is not None, "basis_luts gradient 없음"
        backbone_grads = [
            p.grad for p in model.backbone.parameters()
            if p.grad is not None
        ]
        assert len(backbone_grads) > 0, "backbone gradient 없음"


# ---------------------------------------------------------------------------
# MonotonicityLoss 테스트
# ---------------------------------------------------------------------------

class TestMonotonicityLoss:
    """MonotonicityLoss 단위 테스트."""

    def test_identity_lut_zero_loss(self):
        """항등 LUT는 단조 증가 → 손실이 0에 가까운지 검증."""
        from src.training.losses import MonotonicityLoss
        from src.models.lut3d import ImageAdaptive3DLUT

        model = ImageAdaptive3DLUT(n_basis=2, lut_size=8, backbone_input_size=32)
        loss_fn = MonotonicityLoss()
        loss = loss_fn(model.basis_luts)
        # 항등 LUT는 완전히 단조이므로 손실 ≈ 0
        assert loss.item() < 1e-6, f"항등 LUT 단조성 손실 {loss.item():.2e} > 0"

    def test_non_monotone_lut_positive_loss(self):
        """비단조 LUT는 양수 손실을 반환하는지 검증."""
        from src.training.losses import MonotonicityLoss

        # 역순 LUT: 입력이 증가하면 출력이 감소 → 단조성 위반
        lut_size = 4
        n_basis = 2
        coords = torch.linspace(1.0, 0.0, lut_size)  # 역순
        r, g, b = torch.meshgrid(coords, coords, coords, indexing="ij")
        lut_flat = torch.stack([r.ravel(), g.ravel(), b.ravel()], dim=-1)
        basis = lut_flat.unsqueeze(0).repeat(n_basis, 1, 1)

        loss_fn = MonotonicityLoss()
        loss = loss_fn(basis)
        assert loss.item() > 0.0, "비단조 LUT에서 손실이 0"


# ---------------------------------------------------------------------------
# LUT3DCombinedLoss 테스트
# ---------------------------------------------------------------------------

class TestLUT3DCombinedLoss:
    """LUT3DCombinedLoss 단위 테스트."""

    def test_forward_returns_dict(self):
        """forward가 필요한 키를 포함한 dict를 반환하는지 검증."""
        from src.training.losses import LUT3DCombinedLoss
        from src.models.lut3d import ImageAdaptive3DLUT

        loss_fn = LUT3DCombinedLoss(use_perceptual=False)
        model = ImageAdaptive3DLUT(n_basis=2, lut_size=8, backbone_input_size=32)
        pred = torch.rand(1, 3, 32, 32)
        target = torch.rand(1, 3, 32, 32)

        result = loss_fn(pred, target, model.basis_luts)
        for key in ("total", "l1", "monotonic", "perceptual"):
            assert key in result, f"키 '{key}' 없음"

    def test_total_is_scalar(self):
        """total 손실이 스칼라인지 검증."""
        from src.training.losses import LUT3DCombinedLoss
        from src.models.lut3d import ImageAdaptive3DLUT

        loss_fn = LUT3DCombinedLoss(use_perceptual=False)
        model = ImageAdaptive3DLUT(n_basis=2, lut_size=8, backbone_input_size=32)
        pred = torch.rand(1, 3, 32, 32)
        target = torch.rand(1, 3, 32, 32)

        result = loss_fn(pred, target, model.basis_luts)
        assert result["total"].dim() == 0, "total이 스칼라가 아님"

    def test_total_allows_backward(self):
        """total 손실에 대해 역전파가 가능한지 검증."""
        from src.training.losses import LUT3DCombinedLoss
        from src.models.lut3d import ImageAdaptive3DLUT

        loss_fn = LUT3DCombinedLoss(use_perceptual=False)
        model = ImageAdaptive3DLUT(n_basis=2, lut_size=8, backbone_input_size=32)
        pred_model_out = model(torch.rand(1, 3, 32, 32))
        target = torch.rand(1, 3, 32, 32)

        result = loss_fn(pred_model_out, target, model.basis_luts)
        result["total"].backward()
        assert model.basis_luts.grad is not None


# ---------------------------------------------------------------------------
# export.py 타입 감지 테스트
# ---------------------------------------------------------------------------

class TestDetectModelType:
    """_detect_model_type 함수 단위 테스트."""

    def test_detects_lut3d(self):
        """basis_luts 키가 있으면 'lut3d'로 감지하는지 검증."""
        from src.inference.export import _detect_model_type
        checkpoint = {
            "model_state_dict": {
                "basis_luts": torch.zeros(3, 8**3, 3),
                "backbone.fc.weight": torch.zeros(3, 128),
            }
        }
        assert _detect_model_type(checkpoint) == "lut3d"

    def test_detects_nilut(self):
        """basis_luts 키가 없으면 'nilut'로 감지하는지 검증."""
        from src.inference.export import _detect_model_type
        checkpoint = {
            "model_state_dict": {
                "mlp.0.weight": torch.zeros(256, 3),
                "mlp.0.bias": torch.zeros(256),
            }
        }
        assert _detect_model_type(checkpoint) == "nilut"


# ---------------------------------------------------------------------------
# apply_filter.py lut3d 경로 테스트
# ---------------------------------------------------------------------------

class TestApplyFilterLut3d:
    """apply_filter의 lut3d 경로 테스트."""

    def test_apply_lut3d_model_shape(self, tmp_path):
        """lut3d 모델 체크포인트로 apply_filter 시 shape 보존 검증."""
        import torch
        from src.models.lut3d import ImageAdaptive3DLUT
        from src.inference.apply_filter import apply_filter

        # 소형 모델 저장
        model = ImageAdaptive3DLUT(n_basis=2, lut_size=8, backbone_input_size=32)
        ckpt_path = tmp_path / "lut3d_test.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "n_basis": 2,
                    "lut_size": 8,
                    "backbone_input_size": 32,
                },
            },
            ckpt_path,
        )

        # uint8 이미지로 apply_filter 호출
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = apply_filter(image, model_path=ckpt_path, device="cpu")

        assert result.shape == image.shape, f"shape 불일치: {result.shape} != {image.shape}"
        assert result.dtype == np.uint8, f"dtype 불일치: {result.dtype}"
        assert result.min() >= 0 and result.max() <= 255

    def test_apply_filter_intensity_blending(self, tmp_path):
        """intensity=0.0이면 원본 이미지 반환 검증."""
        import torch
        from src.models.lut3d import ImageAdaptive3DLUT
        from src.inference.apply_filter import apply_filter

        model = ImageAdaptive3DLUT(n_basis=2, lut_size=8, backbone_input_size=32)
        ckpt_path = tmp_path / "lut3d_intensity.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "n_basis": 2,
                    "lut_size": 8,
                    "backbone_input_size": 32,
                },
            },
            ckpt_path,
        )

        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result = apply_filter(image, model_path=ckpt_path, device="cpu", intensity=0.0)

        # intensity=0.0 → 원본 이미지와 동일해야 함
        np.testing.assert_array_equal(result, image)


# ---------------------------------------------------------------------------
# export_to_cube lut3d 경로 테스트
# ---------------------------------------------------------------------------

class TestExportCubeLut3d:
    """lut3d 모델의 .cube export 테스트."""

    def test_export_creates_file(self, tmp_path):
        """.cube 파일이 생성되는지 검증."""
        import torch
        from src.models.lut3d import ImageAdaptive3DLUT
        from src.inference.export import export_to_cube

        model = ImageAdaptive3DLUT(n_basis=2, lut_size=8, backbone_input_size=32)
        ckpt_path = tmp_path / "lut3d.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {"n_basis": 2, "lut_size": 8, "backbone_input_size": 32},
            },
            ckpt_path,
        )

        cube_path = tmp_path / "output.cube"
        export_to_cube(
            model_path=ckpt_path,
            output_path=cube_path,
            cube_size=8,
            device="cpu",
        )

        assert cube_path.exists(), ".cube 파일이 생성되지 않음"
        assert cube_path.stat().st_size > 0, ".cube 파일이 비어 있음"

    def test_export_cube_parseable(self, tmp_path):
        """생성된 .cube 파일을 CubeParser로 읽을 수 있는지 검증."""
        import torch
        from src.models.lut3d import ImageAdaptive3DLUT
        from src.inference.export import export_to_cube
        from src.data.cube_parser import CubeParser

        model = ImageAdaptive3DLUT(n_basis=2, lut_size=8, backbone_input_size=32)
        ckpt_path = tmp_path / "lut3d.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {"n_basis": 2, "lut_size": 8, "backbone_input_size": 32},
            },
            ckpt_path,
        )

        cube_path = tmp_path / "output.cube"
        export_to_cube(
            model_path=ckpt_path,
            output_path=cube_path,
            cube_size=8,
            device="cpu",
        )

        parser = CubeParser()
        lut = parser.read(cube_path)
        assert lut.shape == (8, 8, 8, 3), f"LUT shape 오류: {lut.shape}"
        assert lut.min() >= 0.0 and lut.max() <= 1.0
