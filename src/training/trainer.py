"""학습 루프 관리.

학습, 검증, 테스트 루프와 체크포인트 관리를 담당한다.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """모델 학습 관리자.

    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더 (None이면 검증 생략)
        optimizer: 옵티마이저
        criterion: 손실 함수 (forward(pred, target, model) 또는 forward(pred, target) 모두 지원)
        device: 학습 디바이스 ('cpu', 'cuda', 'mps')
        checkpoint_dir: 체크포인트 저장 디렉토리
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
        checkpoint_dir: str | Path = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.current_epoch: int = 0
        self.best_val_loss: float = float("inf")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _criterion_forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """손실 함수 호출 — CombinedLoss와 단순 손실 모두 지원."""
        import inspect

        sig = inspect.signature(self.criterion.forward)
        params = list(sig.parameters.keys())

        if "model" in params:
            # CombinedLoss 스타일: (pred, target, model)
            return self.criterion(pred, target, self.model)
        else:
            # 단순 손실: (pred, target)
            loss = self.criterion(pred, target)
            return {"total": loss}

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """1 에포크 학습.

        Args:
            epoch: 현재 에포크 번호 (로깅용)

        Returns:
            학습 메트릭 딕셔너리 {'loss': ..., 'delta_e': ...}
        """
        self.model.train()

        total_loss = 0.0
        total_delta_e = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in self.train_loader:
            # 데이터 로드
            inp = batch.get("input_rgb", batch.get("input")).to(self.device)
            tgt = batch.get("target_rgb", batch.get("target")).to(self.device)

            # 스타일 인덱스 (CubeDataset 사용 시)
            style_idx = batch.get("style_idx")
            if style_idx is not None:
                style_idx = style_idx.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            pred = self.model(inp, style_idx=style_idx)

            # Loss
            loss_dict = self._criterion_forward(pred, tgt)
            loss = loss_dict["total"]

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            if "delta_e" in loss_dict:
                total_delta_e += loss_dict["delta_e"].item()
            n_batches += 1

        elapsed = time.time() - t0
        avg_loss = total_loss / max(1, n_batches)
        avg_de = total_delta_e / max(1, n_batches) if total_delta_e > 0 else 0.0

        logger.info(
            f"Epoch {epoch:04d} | train_loss={avg_loss:.4f} | "
            f"delta_e={avg_de:.4f} | {elapsed:.1f}s"
        )

        return {"loss": avg_loss, "delta_e": avg_de}

    def validate(self) -> dict[str, float]:
        """검증 수행.

        Returns:
            검증 메트릭 딕셔너리 {'val_loss': ..., 'val_delta_e': ...}
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_delta_e = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inp = batch.get("input_rgb", batch.get("input")).to(self.device)
                tgt = batch.get("target_rgb", batch.get("target")).to(self.device)
                style_idx = batch.get("style_idx")
                if style_idx is not None:
                    style_idx = style_idx.to(self.device)

                pred = self.model(inp, style_idx=style_idx)
                loss_dict = self._criterion_forward(pred, tgt)

                total_loss += loss_dict["total"].item()
                if "delta_e" in loss_dict:
                    total_delta_e += loss_dict["delta_e"].item()
                n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_de = total_delta_e / max(1, n_batches) if total_delta_e > 0 else 0.0

        logger.info(
            f"  Validation | val_loss={avg_loss:.4f} | val_delta_e={avg_de:.4f}"
        )

        return {"val_loss": avg_loss, "val_delta_e": avg_de}

    def fit(self, epochs: int, early_stopping_patience: int = 10) -> None:
        """전체 학습 실행.

        Args:
            epochs: 총 에포크 수
            early_stopping_patience: 검증 손실이 개선되지 않으면 조기 종료할 에포크 수
        """
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(epoch)

            val_metrics = self.validate()

            # 스케줄러 스텝
            if self.scheduler is not None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau

                if isinstance(self.scheduler, ReduceLROnPlateau):
                    val_loss = val_metrics.get("val_loss", train_metrics["loss"])
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Best model 저장
            current_loss = val_metrics.get("val_loss", train_metrics["loss"])
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                self.save_checkpoint(self.checkpoint_dir / "best.pt", is_best=True)
                patience_counter = 0
                logger.info(f"  -> Best model saved (loss={current_loss:.4f})")
            else:
                patience_counter += 1

            # 주기적 체크포인트 저장 (10 에포크마다)
            if epoch % 10 == 0:
                self.save_checkpoint(self.checkpoint_dir / f"epoch_{epoch:04d}.pt")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(
                    f"Early stopping: {early_stopping_patience} 에포크 동안 개선 없음 "
                    f"(epoch {epoch})"
                )
                break

    def save_checkpoint(self, path: str | Path, is_best: bool = False) -> None:
        """체크포인트 저장.

        Args:
            path: 저장 경로
            is_best: 최고 성능 모델 여부 (로깅용)
        """
        path = Path(path)
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "model_config": {
                # NILUT 모델인 경우 재구성에 필요한 설정 저장
                "num_styles": getattr(self.model, "num_styles", None),
                "style_dim": getattr(self.model, "style_dim", 0),
            },
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """체크포인트 로드.

        Args:
            path: 체크포인트 경로

        Raises:
            FileNotFoundError: 체크포인트 파일이 없을 때
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없음: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(
            f"체크포인트 로드 완료: {path} "
            f"(epoch={self.current_epoch}, best_val_loss={self.best_val_loss:.4f})"
        )
