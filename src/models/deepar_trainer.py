"""DeepAR model training pipeline with early stopping, checkpointing, and logging.

Executes chronological training and validation, ensuring no future information
leaks into the training process.
"""
from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.deepar_config import DeepARConfig
from src.models.deepar_dataset import DeepARTrainingDataset
from src.models.deepar_model import DeepARNetwork, get_device

logger = logging.getLogger(__name__)


class DeepARTrainer:
    """Trainer for DeepAR neural network."""

    def __init__(
        self,
        model: DeepARNetwork,
        config: DeepARConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or get_device(config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        )

        self.history: List[Dict[str, Any]] = []
        self.best_val_loss: float = float("inf")
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch: int = 0
        self.total_train_time: float = 0.0

    def fit(
        self,
        train_dataset: DeepARTrainingDataset,
        val_dataset: DeepARTrainingDataset,
        checkpoint_dir: Optional[Path | str] = None,
    ) -> DeepARTrainer:
        """Execute full training loop across configured epochs."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=len(train_dataset) > self.config.batch_size,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        start_time = perf_counter()
        patience_counter = 0

        logger.info(
            f"Starting DeepAR training on {self.device}: "
            f"{len(train_dataset)} train samples, {len(val_dataset)} val samples, "
            f"batch size {self.config.batch_size}, epochs {self.config.epochs}"
        )

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = perf_counter()

            # --- Training phase ---
            self.model.train()
            train_losses = []

            for batch in train_loader:
                target = batch["target"].to(self.device)
                covariates = batch["covariates"].to(self.device)
                static_cat = batch["static_cat"].to(self.device)
                scale = batch["scale"].to(self.device)

                self.optimizer.zero_grad()
                loss, _ = self.model(target, covariates, static_cat, scale)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Epoch {epoch}: NaN/Inf training loss detected; skipping batch")
                    continue

                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )

                self.optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

            # --- Validation phase ---
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    target = batch["target"].to(self.device)
                    covariates = batch["covariates"].to(self.device)
                    static_cat = batch["static_cat"].to(self.device)
                    scale = batch["scale"].to(self.device)

                    v_loss, _ = self.model(target, covariates, static_cat, scale)
                    if not (torch.isnan(v_loss) or torch.isinf(v_loss)):
                        val_losses.append(v_loss.item())

            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            epoch_duration = perf_counter() - epoch_start

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "duration_seconds": epoch_duration,
            })

            # Check for best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_state_dict = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1 or epoch == self.config.epochs:
                logger.info(
                    f"Epoch {epoch:2d}/{self.config.epochs:2d} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.6f} | Time: {epoch_duration:.2f}s"
                )

            # Early stopping check
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(patience={self.config.early_stopping_patience}, best_epoch={self.best_epoch})"
                )
                break

        self.total_train_time = perf_counter() - start_time

        # Load best weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            self.model.to(self.device)

        # Save checkpoint if requested
        if checkpoint_dir is not None:
            self.save_checkpoint(checkpoint_dir)

        return self

    def save_checkpoint(self, path: Path | str) -> Path:
        """Save training checkpoint to disk."""
        target_path = Path(path)
        if target_path.is_dir() or target_path.suffix == "":
            target_path.mkdir(parents=True, exist_ok=True)
            ckpt_file = target_path / "deepar_checkpoint.pt"
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_file = target_path

        checkpoint = {
            "model_state_dict": self.best_state_dict or self.model.state_dict(),
            "config": self.config.to_dict(),
            "vocab_sizes": self.model.vocab_sizes,
            "covariate_dim": self.model.covariate_dim,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "total_train_time": self.total_train_time,
            "num_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "device": str(self.device),
        }
        torch.save(checkpoint, ckpt_file)
        logger.info(f"Saved DeepAR checkpoint to {ckpt_file}")
        return ckpt_file

    @classmethod
    def load_checkpoint(
        cls,
        path: Path | str,
        device: Optional[torch.device] = None,
    ) -> Tuple[DeepARNetwork, DeepARConfig, Dict[str, Any]]:
        """Load DeepAR model and config from checkpoint."""
        ckpt_path = Path(path)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "deepar_checkpoint.pt"

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = DeepARConfig.from_dict(checkpoint["config"])
        target_device = device or get_device(config.device)

        model = DeepARNetwork(
            config=config,
            vocab_sizes=checkpoint["vocab_sizes"],
            covariate_dim=checkpoint["covariate_dim"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(target_device)
        model.eval()

        metadata = {
            "best_epoch": checkpoint.get("best_epoch", 0),
            "best_val_loss": checkpoint.get("best_val_loss", float("nan")),
            "history": checkpoint.get("history", []),
            "total_train_time": checkpoint.get("total_train_time", 0.0),
            "num_parameters": checkpoint.get("num_parameters", 0),
            "device": str(target_device),
        }
        return model, config, metadata

