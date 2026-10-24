"""Pre-registered synthetic seasonal gate for the M7 state-space variants.

Data-generating process, fixed before execution: 24 independent replications
from seed 20261022, each with 364 training days and 28 held-out days.  Counts
are Poisson with expected demand ``3 * multiplier[weekday]``, where the fixed
weekly multipliers are [0.5, 0.7, 0.9, 1.2, 1.6, 1.4, 0.8].  We score the
mean MAE against held-out simulated counts (the registered metric), while also
reporting MAE against the known expected value.  The pre-registered pass rule
is that BOTH seasonal variants' count MAE must be at least 10% lower than
local_level's.  This rule must not be loosened after observing results.
"""
from time import perf_counter

import numpy as np
import pandas as pd

from src.models.kalman_forecaster import VARIANTS, _fit_one_series, forecast


N_REPLICATIONS = 24
TRAIN_DAYS = 364
HORIZON = 28
SEED = 20261022
MULTIPLIERS = np.array([0.5, 0.7, 0.9, 1.2, 1.6, 1.4, 0.8])
BASE_LEVEL = 3.0
SEASONAL_MARGIN = 0.10


def _series_frame(values: np.ndarray) -> pd.DataFrame:
    days = np.arange(1, len(values) + 1)
    return pd.DataFrame({"day_num": days, "wday": ((days - 1) % 7) + 1, "units_sold": values})


def synthetic_seasonal_results() -> pd.DataFrame:
    """Run the registered DGP and return one mean-MAE row per variant."""
    rng = np.random.default_rng(SEED)
    rows = []
    expected_cycle = BASE_LEVEL * MULTIPLIERS
    for replication in range(N_REPLICATIONS):
        weekdays = np.arange(TRAIN_DAYS + HORIZON) % 7
        expected = expected_cycle[weekdays]
        counts = rng.poisson(expected).astype(float)
        history, held_out = counts[:TRAIN_DAYS], counts[TRAIN_DAYS:]
        expected_future = expected[TRAIN_DAYS:]
        frame = _series_frame(history)
        for variant in VARIANTS:
            fitted = _fit_one_series(f"synthetic_{replication}", frame, variant, None, 0.98)
            output = forecast(fitted.state, fitted.covariance, fitted.T, fitted.Z, fitted.Q, fitted.R, HORIZON)
            prediction = output["y_pred"] * fitted.scale
            rows.append({"replication": replication, "variant": variant,
                         "count_mae": np.mean(np.abs(prediction - held_out)),
                         "expected_value_mae": np.mean(np.abs(prediction - expected_future))})
    return pd.DataFrame(rows).groupby("variant", sort=False)[["count_mae", "expected_value_mae"]].mean().reset_index()


def representative_timing() -> pd.DataFrame:
    """One same-fixture wall-clock fit/predict measurement per variant."""
    rng = np.random.default_rng(SEED)
    days = np.arange(TRAIN_DAYS + HORIZON)
    counts = rng.poisson((BASE_LEVEL * MULTIPLIERS[days % 7])).astype(float)
    frame = _series_frame(counts[:TRAIN_DAYS])
    rows = []
    for variant in VARIANTS:
        started = perf_counter()
        fitted = _fit_one_series("timing_series", frame, variant, None, 0.98)
        fit_seconds = perf_counter() - started
        started = perf_counter()
        forecast(fitted.state, fitted.covariance, fitted.T, fitted.Z, fitted.Q, fitted.R, HORIZON)
        predict_seconds = perf_counter() - started
        rows.append({"variant": variant, "observed_training_days": int(fitted.summary["observed_targets"]),
                     "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                     "converged": fitted.summary["convergence_status"] == "converged",
                     "fallback": bool(fitted.summary["fallback"])})
    return pd.DataFrame(rows)


def test_synthetic_seasonal_gate(capsys):
    results = synthetic_seasonal_results()
    print("Synthetic seasonal gate: mean MAE over 24 fixed-seed replications")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    local_level = results.loc[results.variant.eq("local_level"), "count_mae"].item()
    for seasonal_variant in ("local_level_seasonal", "local_trend_seasonal"):
        seasonal_mae = results.loc[results.variant.eq(seasonal_variant), "count_mae"].item()
        assert seasonal_mae <= local_level * (1 - SEASONAL_MARGIN), (
            f"{seasonal_variant} count MAE={seasonal_mae:.6f}; local_level={local_level:.6f}; "
            f"required <= {local_level * (1 - SEASONAL_MARGIN):.6f} (10% lower)"
        )
