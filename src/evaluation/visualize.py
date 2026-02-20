"""결과 시각화.

원본/스타일/결과 비교 이미지 및 ΔE 히트맵 생성.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from src.evaluation.metrics import compute_delta_e


def create_comparison_grid(
    images: list[np.ndarray],
    labels: list[str],
    output_path: str | Path,
    cols: int = 3,
) -> None:
    """이미지 비교 그리드 생성.

    Args:
        images: 비교할 이미지 리스트, 각 원소 [H, W, 3], float32 또는 uint8
        labels: 각 이미지의 레이블 (images와 같은 길이)
        output_path: 출력 이미지 경로
        cols: 그리드 열 수

    Raises:
        ImportError: matplotlib가 설치되어 있지 않을 때
        ValueError: images와 labels 길이가 다를 때
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib가 필요합니다: pip install matplotlib")

    if len(images) != len(labels):
        raise ValueError(f"images({len(images)})와 labels({len(labels)})의 길이가 다릅니다.")

    n = len(images)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1)  # 1D로 변환

    for i, (img, label) in enumerate(zip(images, labels)):
        ax = axes[i]
        # float32 [0, 1] 또는 uint8 [0, 255] 모두 처리
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    # 빈 서브플롯 숨김
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_delta_e_heatmap(
    image1: np.ndarray,
    image2: np.ndarray,
    output_path: str | Path,
) -> None:
    """두 이미지 간 픽셀별 ΔE 히트맵 생성.

    ΔE < 1.0은 인지적으로 동일, 1~3은 미세 차이, >3은 명확한 차이를 나타낸다.

    Args:
        image1: 첫 번째 이미지 [H, W, 3], float32, 범위 [0, 1]
        image2: 두 번째 이미지 [H, W, 3], float32, 범위 [0, 1]
        output_path: 출력 히트맵 경로

    Raises:
        ImportError: matplotlib가 설치되어 있지 않을 때
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        raise ImportError("matplotlib가 필요합니다: pip install matplotlib")

    from src.utils.color import rgb_to_lab, delta_e_cie76

    image1 = np.asarray(image1, dtype=np.float32)
    image2 = np.asarray(image2, dtype=np.float32)

    lab1 = rgb_to_lab(image1)
    lab2 = rgb_to_lab(image2)
    delta_map = delta_e_cie76(lab1, lab2)  # [H, W]

    avg_de = float(delta_map.mean())
    max_de = float(delta_map.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 원본 이미지
    axes[0].imshow(np.clip(image1, 0, 1))
    axes[0].set_title("Image 1 (Original)")
    axes[0].axis("off")

    # 비교 이미지
    axes[1].imshow(np.clip(image2, 0, 1))
    axes[1].set_title("Image 2 (Processed)")
    axes[1].axis("off")

    # ΔE 히트맵 (최대 10으로 클리핑하여 시각화)
    im = axes[2].imshow(delta_map, cmap="hot", vmin=0, vmax=10)
    axes[2].set_title(f"ΔE Heatmap\nMean={avg_de:.3f}, Max={max_de:.3f}")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f"CIE76 ΔE Analysis (Mean={avg_de:.3f})", fontsize=12)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
