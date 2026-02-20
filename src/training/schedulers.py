"""학습률 스케줄러 유틸리티."""

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    LambdaLR,
    LRScheduler,
)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    epochs: int = 100,
    warmup_epochs: int = 5,
) -> LRScheduler:
    """학습률 스케줄러 생성.

    지원 타입:
        - 'cosine': Linear warmup + Cosine annealing decay
        - 'step': 30 에포크마다 0.1 감소
        - 'plateau': 검증 손실 정체 시 0.5 감소

    Args:
        optimizer: 옵티마이저
        scheduler_type: 스케줄러 종류 ('cosine', 'step', 'plateau')
        epochs: 총 에포크 수
        warmup_epochs: 웜업 에포크 수 (cosine 모드에서만 사용)

    Returns:
        LRScheduler 인스턴스
    """
    if scheduler_type == "cosine":
        # Linear warmup: 0 -> 1.0 (warmup_epochs 동안)
        # Cosine decay: epochs - warmup_epochs 동안 감소
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            import math
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif scheduler_type == "step":
        # 30 에포크마다 학습률을 0.1배로 감소
        return StepLR(optimizer, step_size=30, gamma=0.1)

    elif scheduler_type == "plateau":
        # 검증 손실이 5 에포크 동안 개선되지 않으면 0.5배로 감소
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

    else:
        raise ValueError(
            f"지원하지 않는 스케줄러: {scheduler_type}. "
            f"'cosine', 'step', 'plateau' 중 하나를 선택하세요."
        )
