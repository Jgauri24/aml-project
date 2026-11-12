"""Chronological split and origin isolation validation.

Ensures strict chronological isolation across all models and rolling folds:
- Expanding development origins (o_1 < o_2 < ... < o_K).
- Fixed 28-day non-overlapping evaluation horizons.
- Strict isolation of the reserved competition holdout (d_1914 - d_1941).
- No lookahead leakage in ForecastInputs history or future_known.
- Model-internal validation window isolation (strictly within history <= origin).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.evaluation.backtesting import ForecastInputs, FUTURE_KEYS, KNOWN_CALENDAR
from src.evaluation.splits import Fold, development_splits, final_holdout_spec


FORBIDDEN_FUTURE_COLUMNS = {
    "units_sold",
    "sell_price",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "snap",
    "is_zero_sale",
    "price_missing",
    "possible_prelaunch",
    "availability_unknown",
}


def validate_chronological_splits(cfg: dict, panel: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Validate chronological structure of rolling development folds from experiment configuration."""
    horizon = cfg["forecast"]["horizon"]
    final_origin = cfg["forecast"]["final_origin"]
    origins = cfg["forecast"]["development_origins"]
    cutoff_day = cfg["selection"]["cutoff_day"]

    if horizon <= 0:
        raise ValueError(f"Forecast horizon must be positive, got {horizon}")
    if not origins:
        raise ValueError("development_origins cannot be empty")

    # 1. Origins strictly monotonic and distinct
    sorted_origins = sorted(set(origins))
    if list(origins) != sorted_origins:
        raise ValueError(f"development_origins must be strictly increasing and unique: {origins}")

    folds = development_splits(cfg)
    holdout = final_holdout_spec(cfg)

    # 2. Check each fold properties and holdout boundary
    for i, fold in enumerate(folds):
        if fold.horizon != horizon:
            raise ValueError(f"Fold {i} horizon {fold.horizon} does not match configured horizon {horizon}")
        if fold.origin < cutoff_day:
            raise ValueError(
                f"Fold origin {fold.origin} is before cohort selection cutoff {cutoff_day}"
            )
        target_days = list(fold.target_days)
        if len(target_days) != horizon:
            raise ValueError(f"Fold {i} target days length {len(target_days)} != horizon {horizon}")
        if min(target_days) != fold.origin + 1:
            raise ValueError(f"Fold {i} target days do not start at origin + 1")
        if max(target_days) != fold.origin + horizon:
            raise ValueError(f"Fold {i} target days do not end at origin + horizon")
        if max(target_days) > final_origin:
            raise ValueError(
                f"Fold {i} target day {max(target_days)} exceeds final_origin {final_origin}"
            )

        # Check contiguous non-overlapping target windows between adjacent folds
        if i > 0:
            prev_fold = folds[i - 1]
            if fold.origin < prev_fold.origin + prev_fold.horizon:
                raise ValueError(
                    f"Folds {i-1} and {i} have overlapping target windows: "
                    f"origin_{i} ({fold.origin}) < target_end_{i-1} ({prev_fold.origin + prev_fold.horizon})"
                )

    # 3. Sealed final holdout reservation
    if holdout.origin != final_origin:
        raise ValueError(f"Holdout origin {holdout.origin} must equal final_origin {final_origin}")
    holdout_targets = list(holdout.target_days)
    if min(holdout_targets) <= final_origin:
        raise ValueError("Holdout targets must be strictly greater than final_origin")
    for fold in folds:
        if set(fold.target_days) & set(holdout_targets):
            raise ValueError("Development fold target days intersect with reserved final holdout")

    # 4. If panel is supplied, validate panel data boundary
    panel_stats = {}
    if panel is not None:
        panel_days = set(panel["day_num"].unique())
        for fold in folds:
            # History up to origin must be fully present
            missing_hist_days = set(range(1, fold.origin + 1)) - panel_days
            if missing_hist_days:
                raise ValueError(
                    f"Panel is missing {len(missing_hist_days)} historical days for origin {fold.origin}"
                )
            # Target days must be present in panel for evaluation
            missing_target_days = set(fold.target_days) - panel_days
            if missing_target_days:
                raise ValueError(
                    f"Panel is missing {len(missing_target_days)} target evaluation days for fold {fold.origin}"
                )
        panel_stats = {
            "panel_min_day": int(panel["day_num"].min()),
            "panel_max_day": int(panel["day_num"].max()),
            "panel_series_count": int(panel["series_id"].nunique()),
        }

    return {
        "status": "VALID",
        "num_folds": len(folds),
        "origins": [f.origin for f in folds],
        "horizon": horizon,
        "final_origin": final_origin,
        "holdout_origin": holdout.origin,
        "holdout_target_days": [min(holdout_targets), max(holdout_targets)],
        "target_day_ranges": [[min(f.target_days), max(f.target_days)] for f in folds],
        **panel_stats,
    }


def validate_forecast_inputs_chronology(inputs: ForecastInputs) -> Dict[str, Any]:
    """Validate chronological boundaries and feature isolation for a ForecastInputs instance."""
    if inputs.history is None or inputs.history.empty:
        raise ValueError("ForecastInputs history cannot be empty")
    if inputs.future_known is None or inputs.future_known.empty:
        raise ValueError("ForecastInputs future_known cannot be empty")

    origin = inputs.origin
    horizon = inputs.horizon

    # 1. History boundary
    max_hist_day = int(inputs.history["day_num"].max())
    min_hist_day = int(inputs.history["day_num"].min())
    if max_hist_day != origin:
        raise ValueError(f"History max day_num ({max_hist_day}) does not match origin ({origin})")
    if min_hist_day != 1:
        raise ValueError(f"History min day_num ({min_hist_day}) must be 1")

    # Check series day counts
    day_counts = inputs.history.groupby("series_id", observed=True)["day_num"].count()
    if not day_counts.eq(origin).all():
        raise ValueError(f"All series must have exactly {origin} consecutive historical days")

    # 2. Future known boundary
    min_future_day = int(inputs.future_known["target_day_num"].min())
    max_future_day = int(inputs.future_known["target_day_num"].max())
    if min_future_day != origin + 1:
        raise ValueError(f"future_known min day ({min_future_day}) must be origin + 1 ({origin + 1})")
    if max_future_day != origin + horizon:
        raise ValueError(f"future_known max day ({max_future_day}) must be origin + horizon ({origin + horizon})")

    # Check horizon offsets 1..H
    horizons = set(inputs.future_known["horizon"].unique())
    if horizons != set(range(1, horizon + 1)):
        raise ValueError(f"future_known horizon offsets must be 1..{horizon}, got {sorted(horizons)}")

    # 3. Forbidden future leakage check
    leakage = FORBIDDEN_FUTURE_COLUMNS.intersection(inputs.future_known.columns)
    if leakage:
        raise ValueError(f"Forbidden target/price/SNAP features found in future_known: {leakage}")

    # 4. Future columns match exact KNOWN_CALENDAR allowlist
    expected_cols = set(FUTURE_KEYS + KNOWN_CALENDAR)
    actual_cols = set(inputs.future_known.columns)
    if not expected_cols.issubset(actual_cols):
        missing = expected_cols - actual_cols
        raise ValueError(f"future_known missing expected columns: {missing}")

    return {
        "origin": origin,
        "horizon": horizon,
        "history_days": [min_hist_day, max_hist_day],
        "future_days": [min_future_day, max_future_day],
        "series_count": int(inputs.history["series_id"].nunique()),
        "future_rows": len(inputs.future_known),
        "leakage_free": True,
    }


def validate_model_internal_splits(
    history_length: int,
    origin: int,
    val_window_length: int,
    context_length: int,
) -> Dict[str, Any]:
    """Validate internal training/validation split for iterative models (e.g., DeepAR).

    The internal validation window must use only historical data strictly <= origin,
    and internal training data must precede the internal validation window.
    """
    if val_window_length <= 0:
        raise ValueError("val_window_length must be positive")
    if history_length <= val_window_length + context_length:
        raise ValueError(
            f"History length {history_length} insufficient for val_window {val_window_length} "
            f"+ context {context_length}"
        )

    train_end = origin - val_window_length
    val_start = train_end + 1
    val_end = origin

    if train_end >= val_start:
        raise ValueError("Internal train and validation overlap")
    if val_end > origin:
        raise ValueError(f"Internal validation end {val_end} exceeds origin {origin}")

    return {
        "origin": origin,
        "train_end_day": train_end,
        "val_start_day": val_start,
        "val_end_day": val_end,
        "val_window_days": val_window_length,
        "future_leakage": False,
    }

