"""DeepAR forecaster adhering to the project BaseForecaster contract.

Provides fit(inputs) and predict(inputs) -> pd.DataFrame matching PREDICTION_COLUMNS.
Supports rolling-origin evaluation, autoregressive Monte Carlo inference, and checkpointing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.evaluation.backtesting import (
    BaseForecaster,
    ForecastInputs,
    FUTURE_KEYS,
    PREDICTION_COLUMNS,
    validate_predictions,
)
from src.models.deepar_config import DeepARConfig
from src.models.deepar_dataset import prepare_deepar_data, DeepARPreparedData
from src.models.deepar_model import DeepARNetwork, get_device
from src.models.deepar_trainer import DeepARTrainer

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set global random seed for torch and numpy."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeepARForecaster(BaseForecaster):
    """Global DeepAR probabilistic forecaster for multiple retail time series."""

    MODEL_NAME = "deepar"

    def __init__(
        self,
        config: Optional[DeepARConfig] = None,
        checkpoint_path: Optional[Path | str] = None,
    ) -> None:
        self.config = config or DeepARConfig()
        self.config.validate()
        self.checkpoint_path = checkpoint_path

        self.model_: Optional[DeepARNetwork] = None
        self.trainer_: Optional[DeepARTrainer] = None
        self.prepared_data_: Optional[DeepARPreparedData] = None
        self.fitted_origin_: Optional[int] = None
        self.fit_summary_: Optional[pd.DataFrame] = None
        self.sample_trajectories_: Optional[np.ndarray] = None

    def fit(self, inputs: ForecastInputs) -> DeepARForecaster:
        """Fit DeepAR global recurrent network on available history up to origin."""
        set_seed(self.config.seed)
        start_time = perf_counter()

        logger.info(
            f"Fitting DeepAR for origin d_{inputs.origin}: "
            f"{inputs.history['series_id'].nunique()} series, context={self.config.context_length}, "
            f"pred_len={self.config.prediction_length}, epochs={self.config.epochs}"
        )

        # 1. Prepare data with strict chronological separation
        self.prepared_data_ = prepare_deepar_data(inputs, self.config)

        # 2. Instantiate model
        vocab_sizes = {
            col: self.prepared_data_.encoder.vocab_sizes_[col]
            for col in self.prepared_data_.encoder.cat_columns
        }
        self.model_ = DeepARNetwork(
            config=self.config,
            vocab_sizes=vocab_sizes,
            covariate_dim=self.prepared_data_.covariate_dim,
        )

        # 3. Train model
        device = get_device(self.config.device)
        self.trainer_ = DeepARTrainer(self.model_, self.config, device=device)
        self.trainer_.fit(
            train_dataset=self.prepared_data_.train_dataset,
            val_dataset=self.prepared_data_.val_dataset,
            checkpoint_dir=self.checkpoint_path,
        )

        self.fitted_origin_ = inputs.origin
        fit_seconds = perf_counter() - start_time

        # Record fit summary
        n_params = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        self.fit_summary_ = pd.DataFrame([{
            "model": self.MODEL_NAME,
            "origin": inputs.origin,
            "n_series": len(self.prepared_data_.series_ids),
            "train_samples": len(self.prepared_data_.train_dataset),
            "val_samples": len(self.prepared_data_.val_dataset),
            "best_epoch": self.trainer_.best_epoch,
            "best_val_loss": self.trainer_.best_val_loss,
            "trainable_params": n_params,
            "fit_seconds": fit_seconds,
            "device": str(device),
            "distribution": self.config.distribution,
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "context_length": self.config.context_length,
        }])

        logger.info(
            f"DeepAR fit completed in {fit_seconds:.2f}s "
            f"(best val loss: {self.trainer_.best_val_loss:.4f} at epoch {self.trainer_.best_epoch})"
        )
        return self

    def predict(self, inputs: ForecastInputs) -> pd.DataFrame:
        """Generate autoregressive probabilistic forecasts over the forecast horizon."""
        if self.model_ is None or self.fitted_origin_ is None:
            raise RuntimeError("Call fit before predict")
        if inputs.origin != self.fitted_origin_:
            raise RuntimeError(
                f"Model was fitted for origin d_{self.fitted_origin_}, but predict requested d_{inputs.origin}. "
                "Fit a fresh forecaster per origin."
            )

        set_seed(self.config.seed)
        start_time = perf_counter()
        device = get_device(self.config.device)
        self.model_.to(device)
        self.model_.eval()

        inf_data = self.prepared_data_.inference_data
        context_targets = inf_data["context_targets"].to(device)
        context_covs = inf_data["context_covs"].to(device)
        future_covs = inf_data["future_covs"].to(device)
        static_cats = inf_data["static_cats"].to(device)
        scales = inf_data["scales"].to(device)

        # Generate sample trajectories: shape (n_series, num_samples, horizon)
        raw_samples = self.model_.forecast(
            context_targets=context_targets,
            context_covs=context_covs,
            future_covs=future_covs,
            static_cats=static_cats,
            scales=scales,
            num_samples=self.config.num_samples,
        )

        samples_np = raw_samples.cpu().numpy()  # (n_series, num_samples, horizon)
        self.sample_trajectories_ = samples_np

        n_series, n_samples, horizon = samples_np.shape
        series_ids = self.prepared_data_.series_ids

        # Compute empirical summary statistics
        # Mean across samples
        mean_forecast = np.mean(samples_np, axis=1)  # (n_series, horizon)
        # Quantiles across samples
        q10 = np.quantile(samples_np, 0.10, axis=1)
        q50 = np.quantile(samples_np, 0.50, axis=1)
        q90 = np.quantile(samples_np, 0.90, axis=1)

        # Enforce valid quantile ordering: 0 <= q10 <= q50 <= q90
        q10 = np.maximum(0.0, q10)
        q50 = np.maximum(q10, q50)
        q90 = np.maximum(q50, q90)

        # Format rows aligned to future_known grid
        future_grid = inputs.future_known[FUTURE_KEYS].copy()

        # Build dataframe of predictions
        rows = []
        for s_idx, sid in enumerate(series_ids):
            for h in range(1, horizon + 1):
                rows.append({
                    "series_id": sid,
                    "origin": inputs.origin,
                    "horizon": h,
                    "forecast_mean": float(mean_forecast[s_idx, h - 1]),
                    "y_pred": float(q50[s_idx, h - 1]),
                    "q10": float(q10[s_idx, h - 1]),
                    "q50": float(q50[s_idx, h - 1]),
                    "q90": float(q90[s_idx, h - 1]),
                })

        pred_df = pd.DataFrame(rows)
        # Merge onto future_grid to preserve exact key ordering and timestamps
        result = future_grid.merge(pred_df, on=["series_id", "origin", "horizon"], how="left")
        result["model"] = self.MODEL_NAME

        # Verify columns and nonnegativity
        validate_predictions(result[PREDICTION_COLUMNS], inputs.future_known, model_name=self.MODEL_NAME)

        predict_seconds = perf_counter() - start_time
        logger.info(
            f"DeepAR predict completed in {predict_seconds:.2f}s "
            f"({len(result)} predictions, {self.config.num_samples} trajectories per series)"
        )
        return result[PREDICTION_COLUMNS]


def make_deepar_factory(
    config: Optional[DeepARConfig] = None,
    checkpoint_dir: Optional[Path | str] = None,
) -> Callable[[], DeepARForecaster]:
    """Factory returning a fresh DeepARForecaster per fold, as required by run_backtest."""
    cfg = config or DeepARConfig()

    def factory() -> DeepARForecaster:
        return DeepARForecaster(config=cfg, checkpoint_path=checkpoint_dir)

    return factory

