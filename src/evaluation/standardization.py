"""Standardization and validation of forecaster prediction outputs.

Enforces the canonical prediction contract across all models:
- Exact columns: ['model', 'series_id', 'origin', 'horizon', 'target_day_num', 'target_date', 'y_pred', 'q10', 'q50', 'q90']
- Non-negative finite point forecasts (y_pred >= 0)
- Valid quantiles: either all NA (point-only models) or ordered (0 <= q10 <= q50 <= q90)
- Exact 1-to-1 match against requested evaluation future grid
- Standardized data types and stable deterministic row ordering
"""
from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import pandas as pd

from src.evaluation.backtesting import FUTURE_KEYS, PREDICTION_COLUMNS, validate_predictions

logger = logging.getLogger(__name__)


def standardize_prediction_output(
    predictions: pd.DataFrame,
    expected_grid: Optional[pd.DataFrame] = None,
    model_name: Optional[str] = None,
    numerical_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Standardize, clean, and validate model prediction DataFrame.

    Parameters:
    - predictions: Raw output DataFrame from a BaseForecaster.
    - expected_grid: Optional future_known DataFrame containing requested grid keys.
    - model_name: Expected model name string (must be uniform across rows).
    - numerical_tolerance: Epsilon tolerance for minor floating-point clipping.

    Returns:
    - Cleaned and strictly validated pd.DataFrame containing PREDICTION_COLUMNS.
    """
    if predictions is None or predictions.empty:
        raise ValueError("Predictions DataFrame cannot be empty")

    missing_cols = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing_cols:
        raise ValueError(f"Predictions missing required columns: {sorted(missing_cols)}")

    df = predictions[PREDICTION_COLUMNS].copy()

    # 1. Standardize types
    df["model"] = df["model"].astype(str)
    df["series_id"] = df["series_id"].astype(str)
    df["origin"] = df["origin"].astype(int)
    df["horizon"] = df["horizon"].astype(int)
    df["target_day_num"] = df["target_day_num"].astype(int)
    df["target_date"] = pd.to_datetime(df["target_date"])

    # 2. Check model name
    unique_models = df["model"].unique()
    if len(unique_models) != 1:
        raise ValueError(f"Predictions must identify exactly one model, got: {unique_models.tolist()}")
    if model_name is not None and unique_models[0] != model_name:
        raise ValueError(f"Model name mismatch: expected '{model_name}', got '{unique_models[0]}'")

    # 3. Numeric predictions cleaning and bounds enforcement
    y_pred = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(y_pred).any() or not np.isfinite(y_pred).all():
        raise ValueError("Point forecasts (y_pred) contain NaN or non-finite values")

    # Minor float numerical underflow tolerance
    negative_mask = y_pred < 0
    if negative_mask.any():
        if (y_pred[negative_mask] < -numerical_tolerance).any():
            min_val = float(y_pred.min())
            raise ValueError(f"Point forecasts contain significantly negative values (min: {min_val})")
        logger.warning("Clipping slight negative floating-point underflow in y_pred to 0.0")
        y_pred = np.maximum(0.0, y_pred)
    df["y_pred"] = y_pred

    # 4. Quantile processing
    quantiles_df = df[["q10", "q50", "q90"]].copy()
    all_na = quantiles_df.isna().all().all()

    if not all_na:
        for q_col in ["q10", "q50", "q90"]:
            q_vals = pd.to_numeric(quantiles_df[q_col], errors="coerce").to_numpy(dtype=np.float64)
            if np.isnan(q_vals).any() or not np.isfinite(q_vals).all():
                raise ValueError(f"Quantile column '{q_col}' contains NaN or non-finite values")
            quantiles_df[q_col] = q_vals

        q10 = quantiles_df["q10"].to_numpy()
        q50 = quantiles_df["q50"].to_numpy()
        q90 = quantiles_df["q90"].to_numpy()

        # Handle minor floating point underflow
        if (q10 < 0).any():
            if (q10 < -numerical_tolerance).any():
                raise ValueError("q10 contains negative values beyond numerical tolerance")
            q10 = np.maximum(0.0, q10)

        # Ensure ordered quantiles with numerical tolerance
        q50 = np.maximum(q10, q50)
        q90 = np.maximum(q50, q90)

        df["q10"] = q10
        df["q50"] = q50
        df["q90"] = q90
    else:
        df["q10"] = np.nan
        df["q50"] = np.nan
        df["q90"] = np.nan

    # 5. Grid matching if expected_grid is supplied
    if expected_grid is not None:
        validate_predictions(df, expected_grid, model_name=unique_models[0])

    # 6. Sort rows deterministically by origin, series_id, horizon
    df = df.sort_values(["origin", "series_id", "horizon"]).reset_index(drop=True)
    return df[PREDICTION_COLUMNS]

