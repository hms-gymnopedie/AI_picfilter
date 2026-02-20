"""학습 실행 스크립트.

Usage:
    python scripts/train.py --config configs/nilut.yaml
    python scripts/train.py --config configs/nilut.yaml --num-styles 128 --lr 5e-4 --device mps
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.nilut import NILUT
from src.training.trainer import Trainer
from src.training.losses import CombinedLoss
from src.training.schedulers import get_scheduler

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱."""
    parser = argparse.ArgumentParser(description="AI_picfilter NILUT 모델 학습")

    # 설정
    parser.add_argument(
        "--config", type=str, default=None, help="설정 파일 경로 (YAML)"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="학습 디바이스 (cpu/cuda/mps/auto)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="이어서 학습할 체크포인트 경로"
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")

    # 데이터
    parser.add_argument(
        "--data-type",
        type=str,
        default="cube",
        choices=["cube", "paired"],
        help="데이터 타입: cube(.cube 파일) 또는 paired(이미지 쌍)",
    )
    parser.add_argument(
        "--cube-dir",
        type=str,
        default=None,
        help=".cube 파일 디렉토리 (data-type=cube 시)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="원본 이미지 디렉토리 (data-type=paired 시)",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="보정 이미지 디렉토리 (data-type=paired 시)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="배치 크기 (cube 모드: 픽셀 수, paired 모드: 이미지 수)",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 워커 수")

    # 모델
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256, 256],
        help="MLP 은닉층 차원 (예: --hidden-dims 256 256 256)",
    )
    parser.add_argument(
        "--activation", type=str, default="gelu", choices=["relu", "gelu"]
    )
    parser.add_argument(
        "--num-styles",
        type=int,
        default=None,
        help="다중 스타일 수. None이면 단일 스타일",
    )
    parser.add_argument("--style-dim", type=int, default=64, help="스타일 임베딩 차원")

    # 학습
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--scheduler", type=str, default="cosine", choices=["cosine", "step", "plateau"]
    )
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--lambda-delta-e", type=float, default=0.5)
    parser.add_argument("--lambda-smooth", type=float, default=0.01)

    # 출력
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/nilut",
        help="체크포인트 저장 디렉토리",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """재현성을 위한 랜덤 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device(requested: str) -> str:
    """디바이스 자동 감지."""
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"자동 감지된 디바이스: {device}")
    return device


def build_dataloaders(args: argparse.Namespace):
    """데이터로더 생성."""
    from torch.utils.data import DataLoader, random_split

    if args.data_type == "cube":
        if args.cube_dir is None:
            raise ValueError("--cube-dir을 지정해야 합니다 (data-type=cube 시)")

        cube_dir = Path(args.cube_dir)
        cube_files = sorted(cube_dir.glob("*.cube"))
        if not cube_files:
            raise FileNotFoundError(f".cube 파일을 찾을 수 없음: {cube_dir}")

        logger.info(f".cube 파일 {len(cube_files)}개 발견")

        from src.data.dataset import CubeDataset

        dataset = CubeDataset(cube_files, samples_per_cube=100_000)

    elif args.data_type == "paired":
        if args.input_dir is None or args.target_dir is None:
            raise ValueError("--input-dir과 --target-dir을 모두 지정해야 합니다")

        from src.data.dataset import ImagePairDataset

        dataset = ImagePairDataset(
            input_dir=args.input_dir,
            target_dir=args.target_dir,
            image_size=(256, 256),
        )

    else:
        raise ValueError(f"지원하지 않는 data-type: {args.data_type}")

    # 8:2 train/val 분할
    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    logger.info(f"학습 샘플: {n_train:,}, 검증 샘플: {n_val:,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def main() -> None:
    """학습 메인 함수."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    # 랜덤 시드
    set_seed(args.seed)
    logger.info(f"랜덤 시드: {args.seed}")

    # 디바이스
    device = detect_device(args.device)

    # 데이터로더
    train_loader, val_loader = build_dataloaders(args)

    # 모델
    model = NILUT(
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        num_styles=args.num_styles,
        style_dim=args.style_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"NILUT 모델 | 파라미터: {n_params:,} | "
        f"크기: {n_params * 4 / 1024:.1f} KB | "
        f"스타일: {args.num_styles or 1}개"
    )

    # 옵티마이저
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 손실 함수
    criterion = CombinedLoss(
        lambda_delta_e=args.lambda_delta_e,
        lambda_smooth=args.lambda_smooth,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # 스케줄러 설정
    trainer.scheduler = get_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )

    # 체크포인트 재개
    if args.resume:
        trainer.load_checkpoint(args.resume)

    logger.info(f"학습 시작: {args.epochs}에포크, 디바이스={device}")
    trainer.fit(
        epochs=args.epochs, early_stopping_patience=args.early_stopping_patience
    )
    logger.info(f"학습 완료. 최고 val_loss={trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
