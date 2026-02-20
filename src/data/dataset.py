"""PyTorch Dataset 클래스.

이미지 쌍(보정 전/후) 및 .cube 파일 기반 학습 데이터셋.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable

try:
    from PIL import Image as PILImage

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from src.data.cube_parser import CubeParser

# 지원 이미지 확장자
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def _load_image(path: Path, size: Optional[tuple[int, int]] = None) -> torch.Tensor:
    """이미지를 float32 텐서 [3, H, W], 범위 [0, 1]로 로드."""
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow가 필요합니다: pip install Pillow")

    img = PILImage.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size[1], size[0]), PILImage.BICUBIC)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
    return torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]


class ImagePairDataset(Dataset):
    """보정 전/후 이미지 쌍 데이터셋.

    input_dir 과 target_dir 에서 같은 파일명을 가진 이미지 쌍을 로드한다.
    Adobe FiveK 등의 페어 데이터셋에 적합하다.

    Args:
        input_dir: 원본(보정 전) 이미지 디렉토리
        target_dir: 보정 후 이미지 디렉토리
        image_size: 리사이즈 크기 (H, W). None이면 원본 크기 유지
        transform: 추가 augmentation 변환 (텐서 입력/출력)
    """

    def __init__(
        self,
        input_dir: str | Path,
        target_dir: str | Path,
        image_size: Optional[tuple[int, int]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size
        self.transform = transform

        # 입력 디렉토리에서 지원 이미지 파일 스캔
        input_files = {
            f.name: f
            for f in self.input_dir.iterdir()
            if f.suffix.lower() in _IMAGE_EXTS
        }

        # target 디렉토리에서 동일한 파일명 매칭
        self.pairs: list[tuple[Path, Path]] = []
        for name, inp_path in sorted(input_files.items()):
            tgt_path = self.target_dir / name
            if tgt_path.exists():
                self.pairs.append((inp_path, tgt_path))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"매칭된 이미지 쌍이 없습니다. "
                f"input_dir={self.input_dir}, target_dir={self.target_dir}"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """데이터 로드.

        Returns:
            dict with keys:
                'input': 원본 이미지 텐서 [3, H, W], 범위 [0, 1]
                'target': 보정 이미지 텐서 [3, H, W], 범위 [0, 1]
                'filename': 파일명 (str)
        """
        inp_path, tgt_path = self.pairs[idx]

        inp = _load_image(inp_path, self.image_size)
        tgt = _load_image(tgt_path, self.image_size)

        if self.transform is not None:
            inp = self.transform(inp)
            tgt = self.transform(tgt)

        return {
            "input": inp,
            "target": tgt,
            "filename": inp_path.name,
        }


class CubeDataset(Dataset):
    """.cube 파일 기반 학습 데이터셋.

    .cube 파일에서 색상 매핑 쌍(입력 RGB -> 출력 RGB)을 추출한다.
    NILUT의 단일/다중 스타일 학습에 사용된다.

    각 .cube 파일이 하나의 스타일을 나타내며,
    파일 인덱스가 스타일 인덱스로 사용된다.

    Args:
        cube_paths: .cube 파일 경로 리스트
        samples_per_cube: 각 .cube 파일에서 샘플링할 색상 쌍 수
    """

    def __init__(
        self,
        cube_paths: list[str | Path],
        samples_per_cube: int = 100_000,
    ) -> None:
        self.samples_per_cube = samples_per_cube
        parser = CubeParser()

        # 각 스타일(cube)의 (입력 RGB, 목표 RGB) 쌍 미리 준비
        # input은 항등 LUT 좌표, target은 .cube의 색상 변환
        self._input_rgbs: list[np.ndarray] = []  # [N, 3]
        self._target_rgbs: list[np.ndarray] = []  # [N, 3]
        self._style_indices: list[np.ndarray] = []  # [N]

        for style_idx, cube_path in enumerate(cube_paths):
            lut = parser.read(cube_path)  # [size, size, size, 3]
            size = lut.shape[0]

            # 항등 LUT 그리드 좌표 (입력값)
            coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
            r, g, b = np.meshgrid(coords, coords, coords, indexing="ij")
            input_flat = np.stack(
                [r.ravel(), g.ravel(), b.ravel()], axis=-1
            )  # [size^3, 3]
            target_flat = lut.reshape(-1, 3)  # [size^3, 3]

            # 지정된 수만큼 랜덤 샘플링
            n_total = input_flat.shape[0]
            n_sample = min(samples_per_cube, n_total)
            indices = np.random.choice(
                n_total, size=n_sample, replace=n_sample > n_total
            )

            self._input_rgbs.append(input_flat[indices])
            self._target_rgbs.append(target_flat[indices])
            self._style_indices.append(
                np.full(n_sample, fill_value=style_idx, dtype=np.int64)
            )

        self._input_all = np.concatenate(self._input_rgbs, axis=0)  # [total, 3]
        self._target_all = np.concatenate(self._target_rgbs, axis=0)  # [total, 3]
        self._style_all = np.concatenate(self._style_indices, axis=0)  # [total]

    def __len__(self) -> int:
        return len(self._input_all)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """색상 매핑 쌍 반환.

        Returns:
            dict with keys:
                'input_rgb': 입력 RGB [3], float32, 범위 [0, 1]
                'target_rgb': 목표 RGB [3], float32, 범위 [0, 1]
                'style_idx': 스타일 인덱스 (long scalar)
        """
        return {
            "input_rgb": torch.from_numpy(self._input_all[idx]),
            "target_rgb": torch.from_numpy(self._target_all[idx]),
            "style_idx": torch.tensor(self._style_all[idx], dtype=torch.long),
        }
