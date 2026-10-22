"""
Tests for Milestone 6: Single-SKU Kalman Filter.

Organisation
------------
Section 1 — kalman_filter  : hand-calculated reference values, missing-obs rules
Section 2 — forecast_local_level : variance growth, constant mean, MIN_PRED_VAR
Section 3 — fit_local_level : degenerate inputs, convergence checks
Section 4 — KalmanSingleSKU adapter : schema, coverage, quantile order, guards
Section 5 — Engine integration : run_backtest on tiny_panel, leakage check

All tests use only the synthetic ``tiny_panel`` / ``tiny_project`` fixtures
from ``tests/conftest.py``; no raw CSV or canonical parquet is needed.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtesting import (
    PREDICTION_COLUMNS,
    construct_inputs,
    run_backtest,
)
from src.evaluation.splits import development_splits
from src.models.kalman_single_sku import (
    MIN_PRED_VAR,
    MODEL_NAME,
    _P0_DIFFUSE,
    _Q_FALLBACK,
    _R_FALLBACK,
    FilterResult,
    KalmanSingleSKU,
    fit_local_level,
    forecast_local_level,
    kalman_filter,
)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — kalman_filter unit tests
# ─────────────────────────────────────────────────────────────────────────────


def test_kalman_filter_two_step_hand_calculation():
    """Hand-verified 2-step filter: Q=R=1, mu0=0, P0=1.

    t=0  (y=2):
        P_pred = P0 + Q = 1 + 1 = 2
        F = 2 + 1 = 3,  K = 2/3,  v = 2
        mu[0] = 0 + (2/3)*2 = 4/3
        P[0]  = (1 - 2/3) * 2 = 2/3        ← uses P_pred, not P_prev

    t=1  (y=4):
        P_pred = 2/3 + 1 = 5/3
        F = 5/3 + 1 = 8/3,  K = (5/3)/(8/3) = 5/8,  v = 4 - 4/3 = 8/3
        mu[1] = 4/3 + (5/8)*(8/3) = 4/3 + 5/3 = 3
        P[1]  = (1 - 5/8) * (5/3) = (3/8)*(5/3) = 5/8
    """
    y = np.array([2.0, 4.0])
    result = kalman_filter(y, Q=1.0, R=1.0, mu0=0.0, P0=1.0)

    assert result.mu_filtered[0] == pytest.approx(4 / 3, rel=1e-10)
    assert result.P_filtered[0] == pytest.approx(2 / 3, rel=1e-10)
    assert result.mu_filtered[1] == pytest.approx(3.0, rel=1e-10)
    assert result.P_filtered[1] == pytest.approx(5 / 8, rel=1e-10)


def test_update_uses_P_pred_not_P_prev():
    """P_{t|t} = (1-K)*P_{t|t-1}; the prior at t, not the posterior from t-1.

    With Q=0.5, R=1, P0=2:

    t=0:  P_pred = 2+0.5=2.5,  K=2.5/3.5=5/7
          P[0]  = (1-5/7)*2.5 = (2/7)*2.5 = 5/7

    t=1:  P_pred = 5/7 + 0.5 = 8.5/7   ← differs from P[0]=5/7 by Q=0.5
          K = (8.5/7)/(8.5/7+1) = 8.5/15.5 = 17/31
          P[1] = (1-17/31)*(8.5/7) = (14/31)*(8.5/7) = 119/217

    Using P[0]=5/7 instead of P_pred=8.5/7 would give (14/31)*(5/7)=70/217 ≠ 119/217.
    """
    result = kalman_filter(np.array([1.0, 1.0]), Q=0.5, R=1.0, mu0=0.0, P0=2.0)
    assert result.P_filtered[0] == pytest.approx(5 / 7, rel=1e-10)
    assert result.P_filtered[1] == pytest.approx(119 / 217, rel=1e-10)


def test_missing_observation_skips_update():
    """NaN at t=1: mu and P carry forward from the prediction step.

    After t=0 (y=2, Q=R=1, mu0=0, P0=1): mu[0]=4/3, P[0]=2/3  (from above).
    t=1 (missing):
        P_pred = 2/3 + 1 = 5/3
        No update → mu[1] = 4/3,  P[1] = 5/3
    """
    y = np.array([2.0, np.nan, 4.0])
    result = kalman_filter(y, Q=1.0, R=1.0, mu0=0.0, P0=1.0)

    assert result.mu_filtered[1] == pytest.approx(4 / 3, rel=1e-10)
    assert result.P_filtered[1] == pytest.approx(5 / 3, rel=1e-10)
    assert np.isnan(result.innovations[1]), "Innovation for missing obs must be NaN"
    assert np.isnan(result.F[1]), "F for missing obs must be NaN"


def test_multiple_missing_steps_accumulate_variance():
    """Each consecutive missing step adds Q to P; mean stays constant."""
    y = np.array([1.0, np.nan, np.nan, np.nan])
    result = kalman_filter(y, Q=0.5, R=1.0, mu0=0.0, P0=1.0)

    # During the gap, variance must grow strictly
    assert result.P_filtered[2] > result.P_filtered[1]
    assert result.P_filtered[3] > result.P_filtered[2]

    # Mean stays constant when no observations arrive
    assert result.mu_filtered[2] == pytest.approx(result.mu_filtered[1])
    assert result.mu_filtered[3] == pytest.approx(result.mu_filtered[2])


def test_all_nan_returns_prior_propagated():
    """All-missing series: filter propagates prior forward; log-lik = 0."""
    y = np.array([np.nan, np.nan, np.nan])
    result = kalman_filter(y, Q=1.0, R=1.0, mu0=5.0, P0=2.0)

    assert result.mu_filtered[0] == pytest.approx(5.0)   # unchanged from mu0
    assert result.P_filtered[0] == pytest.approx(3.0)    # P0 + Q
    assert result.P_filtered[1] == pytest.approx(4.0)    # 3 + Q
    assert result.log_lik == pytest.approx(0.0)           # no observations


def test_log_likelihood_is_negative_and_finite():
    """With typical data (positive variance), the log-lik must be negative."""
    rng = np.random.default_rng(0)
    y = rng.poisson(lam=2.0, size=100).astype(float)
    result = kalman_filter(y, Q=0.5, R=1.0)
    assert np.isfinite(result.log_lik)
    assert result.log_lik < 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — forecast_local_level unit tests
# ─────────────────────────────────────────────────────────────────────────────


def test_forecast_point_is_constant_mu_T():
    """Local-level point forecast is mu_T for every horizon."""
    mu_T = 3.14
    y_hat, _ = forecast_local_level(mu_T=mu_T, P_T=0.5, Q=0.1, R=0.5, horizon=28)
    assert np.allclose(y_hat, mu_T)


def test_predictive_variance_grows_with_horizon():
    """V_h = P_T + h*Q + R must strictly increase with h when Q > 0."""
    _, pred_vars = forecast_local_level(mu_T=1.0, P_T=0.5, Q=0.1, R=0.5, horizon=28)
    assert np.all(np.diff(pred_vars) > 0), "Variance must grow with h when Q > 0"


def test_predictive_variance_constant_when_Q_zero():
    """With Q=0 (no level drift) variance is constant P_T + R."""
    _, pred_vars = forecast_local_level(mu_T=1.0, P_T=0.5, Q=0.0, R=0.5, horizon=28)
    assert np.allclose(pred_vars, 1.0)  # P_T + R = 0.5 + 0.5 = 1.0


def test_predictive_variance_matches_formula():
    """V_h = P_T + h*Q + R for h=1..5."""
    P_T, Q, R = 0.3, 0.2, 0.7
    _, pred_vars = forecast_local_level(mu_T=0.0, P_T=P_T, Q=Q, R=R, horizon=5)
    for h in range(1, 6):
        expected = P_T + h * Q + R
        assert pred_vars[h - 1] == pytest.approx(expected, rel=1e-10)


def test_pred_var_floored_at_min_pred_var():
    """Floor must apply even when Q=R=P_T=0 (degenerate input)."""
    _, pred_vars = forecast_local_level(mu_T=0.0, P_T=0.0, Q=0.0, R=0.0, horizon=5)
    assert np.all(pred_vars >= MIN_PRED_VAR)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — fit_local_level unit tests
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_all_nan_returns_fallback():
    """All-NaN series triggers a RuntimeWarning and returns documented defaults."""
    with pytest.warns(RuntimeWarning, match="no observed"):
        Q, R, mu_T, P_T = fit_local_level(np.full(10, np.nan))

    assert Q == _Q_FALLBACK
    assert R == _R_FALLBACK
    assert mu_T == 0.0
    assert P_T == pytest.approx(_P0_DIFFUSE)


def test_fit_constant_nonzero_series():
    """Constant series converges without error; Q, R, mu_T must be finite and positive."""
    Q, R, mu_T, P_T = fit_local_level(np.ones(200) * 3.5)
    assert np.isfinite(Q) and Q > 0
    assert np.isfinite(R) and R > 0
    assert np.isfinite(mu_T)
    assert np.isfinite(P_T) and P_T >= 0


def test_fit_all_zero_series():
    """All-zero series (very common in M5 sparse cohort) converges without error."""
    Q, R, mu_T, P_T = fit_local_level(np.zeros(200))
    assert np.isfinite(Q) and Q > 0
    assert np.isfinite(R) and R > 0
    assert np.isfinite(mu_T)
    assert P_T >= 0


def test_fit_sparse_intermittent_series():
    """Series with ~90 % zeros and occasional spikes (typical M5 low-band SKU)."""
    rng = np.random.default_rng(42)
    y = np.zeros(400)
    idx = rng.choice(400, size=40, replace=False)
    y[idx] = rng.exponential(scale=2.0, size=40)
    Q, R, mu_T, P_T = fit_local_level(y)
    assert all(np.isfinite(v) and v > 0 for v in [Q, R])
    assert np.isfinite(mu_T) and np.isfinite(P_T)


def test_fit_returns_nonneg_P_T():
    """Filtered posterior variance P_{T|T} must be nonneg for any valid series."""
    rng = np.random.default_rng(7)
    y = rng.poisson(lam=1.5, size=100).astype(float)
    _, _, _, P_T = fit_local_level(y)
    assert P_T >= 0.0


def test_initial_state_mean_skips_leading_nans():
    """_initial_state_mean uses the first ≤28 *non-NaN* values, not positional slicing.

    With 10 leading NaN entries and 30 observed values [1..30], the series has
    40 elements total.  Positional slicing y[:28] yields [NaN×10, 1..18] whose
    nanmean is 9.5; the correct formula skips NaN positions first, yielding
    y[~np.isnan(y)][:28] = [1..28] whose mean is 14.5.  This test confirms the
    helper returns the correct value and acts as a regression guard against
    reverting to the nanmean(y[:28]) pattern.
    """
    from src.models.kalman_single_sku import _initial_state_mean

    y = np.array([np.nan] * 10 + list(range(1, 31)), dtype=float)

    positional_nanmean = float(np.nanmean(y[:28]))   # wrong: 9.5
    correct_mean = float(np.mean(np.arange(1, 29)))   # correct: 14.5

    # Precondition: confirm the two formulas actually differ for this input.
    assert positional_nanmean != pytest.approx(correct_mean), (
        "Precondition failed: formulas must diverge for the leading-NaN case"
    )
    assert _initial_state_mean(y) == pytest.approx(correct_mean, rel=1e-10)


def test_initial_state_mean_no_nans():
    """Without NaN, _initial_state_mean equals nanmean of the first 28 positions."""
    from src.models.kalman_single_sku import _initial_state_mean

    y = np.arange(1.0, 51.0)          # 50 elements, no NaN
    assert _initial_state_mean(y) == pytest.approx(float(np.mean(y[:28])), rel=1e-10)


def test_initial_state_mean_all_nan():
    """All-NaN series returns 0.0 (consistent with fit_local_level fallback)."""
    from src.models.kalman_single_sku import _initial_state_mean

    assert _initial_state_mean(np.full(20, np.nan)) == 0.0


def test_initial_state_mean_fewer_than_28_obs():
    """Short series with fewer than 28 observed values: uses all of them."""
    from src.models.kalman_single_sku import _initial_state_mean

    y = np.array([np.nan, np.nan, 3.0, 5.0, np.nan, 7.0])
    # Non-NaN values: [3.0, 5.0, 7.0] — only 3, all fewer than 28
    assert _initial_state_mean(y) == pytest.approx(5.0, rel=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — KalmanSingleSKU adapter tests
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_returns_self(tiny_panel):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    model = KalmanSingleSKU()
    assert model.fit(inputs) is model


def test_predict_requires_fit(tiny_panel):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    with pytest.raises(RuntimeError, match="fit"):
        KalmanSingleSKU().predict(inputs)


def test_origin_guard_raises_on_mismatch(tiny_panel):
    """predict() on a different origin than fit() must raise RuntimeError."""
    panel, cfg = tiny_panel
    folds = development_splits(cfg)
    first = construct_inputs(panel, folds[0])
    second = construct_inputs(panel, folds[1])
    model = KalmanSingleSKU()
    model.fit(first)
    with pytest.raises(RuntimeError, match="origin"):
        model.predict(second)


def test_predict_schema(tiny_panel):
    """Output must have exactly PREDICTION_COLUMNS in the correct order."""
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    pred = KalmanSingleSKU().fit(inputs).predict(inputs)
    assert list(pred.columns) == PREDICTION_COLUMNS


def test_predict_coverage_and_uniqueness(tiny_panel):
    """Every requested (series_id, horizon) pair appears exactly once."""
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    pred = KalmanSingleSKU().fit(inputs).predict(inputs)
    assert len(pred) == len(inputs.future_known)
    assert not pred.duplicated(["series_id", "horizon"]).any()


def test_model_name_consistent_across_rows(tiny_panel):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    pred = KalmanSingleSKU().fit(inputs).predict(inputs)
    assert (pred["model"] == MODEL_NAME).all()


def test_point_forecasts_finite_and_nonneg(tiny_panel):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    pred = KalmanSingleSKU().fit(inputs).predict(inputs)
    assert pred["y_pred"].notna().all()
    assert np.isfinite(pred["y_pred"].to_numpy()).all()
    assert (pred["y_pred"] >= 0.0).all()


def test_quantiles_present_finite_nonneg_and_ordered(tiny_panel):
    """All quantiles filled, non-negative, and ordered q10 <= q50 <= q90."""
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    pred = KalmanSingleSKU().fit(inputs).predict(inputs)

    for col in ["q10", "q50", "q90"]:
        assert pred[col].notna().all(), f"{col} contains NaN"
        assert np.isfinite(pred[col].to_numpy()).all(), f"{col} contains non-finite"
        assert (pred[col] >= 0.0).all(), f"{col} contains negative value"

    assert (pred["q10"] <= pred["q50"]).all(), "q10 > q50 detected"
    assert (pred["q50"] <= pred["q90"]).all(), "q50 > q90 detected"


def test_interval_widens_with_horizon(tiny_panel):
    """Predictive interval width at h=max must be >= width at h=1 (Q >= 0)."""
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    pred = KalmanSingleSKU().fit(inputs).predict(inputs)

    for sid, grp in pred.groupby("series_id"):
        grp = grp.sort_values("horizon")
        widths = (grp["q90"] - grp["q10"]).to_numpy()
        assert widths[-1] >= widths[0] - 1e-9, (
            f"Interval narrowed with horizon for series {sid}"
        )


def test_all_nan_series_still_produces_valid_output(tiny_panel):
    """A series with all-NaN units_sold must not cause predict() to fail."""
    panel, cfg = tiny_panel
    panel_nan = panel.copy()
    first_sid = panel_nan["series_id"].cat.categories[0]
    panel_nan.loc[panel_nan["series_id"] == first_sid, "units_sold"] = np.nan

    inputs = construct_inputs(panel_nan, development_splits(cfg)[0])
    with pytest.warns(RuntimeWarning):
        pred = KalmanSingleSKU().fit(inputs).predict(inputs)

    assert pred["y_pred"].notna().all()
    assert (pred["y_pred"] >= 0.0).all()
    assert (pred["q10"] <= pred["q50"]).all()
    assert (pred["q50"] <= pred["q90"]).all()


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Engine integration and leakage tests
# ─────────────────────────────────────────────────────────────────────────────


def test_leakage_future_sales_mutation_does_not_change_predictions(tiny_panel):
    """Corrupting future actuals (post-origin) must not alter predictions.

    The engine's construct_inputs() already enforces the column allowlist and
    the day_num <= origin bound; this test confirms our adapter respects it.
    """
    panel, cfg = tiny_panel
    first_fold = development_splits(cfg)[0]

    corrupted = panel.copy()
    corrupted.loc[corrupted["day_num"] > first_fold.origin, "units_sold"] = 999.0

    inputs_clean = construct_inputs(panel, first_fold)
    inputs_corrupt = construct_inputs(corrupted, first_fold)

    # Verify the engine's boundary enforcement (histories must be identical)
    pd.testing.assert_frame_equal(
        inputs_clean.history.reset_index(drop=True),
        inputs_corrupt.history.reset_index(drop=True),
    )

    # Predictions derived from identical histories must be identical
    pred_clean = KalmanSingleSKU(seed=42).fit(inputs_clean).predict(inputs_clean)
    pred_corrupt = KalmanSingleSKU(seed=42).fit(inputs_corrupt).predict(inputs_corrupt)
    pd.testing.assert_frame_equal(pred_clean, pred_corrupt)


def test_full_engine_integration_on_tiny_panel(tiny_panel, tmp_path):
    """run_backtest completes; output schema and holdout guard verified."""
    panel, cfg = tiny_panel
    predictions, metrics, metadata = run_backtest(
        panel, cfg, partial(KalmanSingleSKU, seed=42), output_dir=str(tmp_path)
    )

    # Schema
    assert list(predictions.columns) == PREDICTION_COLUMNS

    # tiny_panel: 3 series × 2 folds × 3 horizon = 18 rows
    assert len(predictions) == 18, f"Expected 18 rows, got {len(predictions)}"

    # Point forecasts valid
    assert predictions["y_pred"].ge(0).all()
    assert np.isfinite(predictions["y_pred"].to_numpy()).all()

    # Quantiles populated (Kalman provides genuine uncertainty)
    assert predictions[["q10", "q50", "q90"]].notna().all().all()
    assert (predictions["q10"] <= predictions["q50"]).all()
    assert (predictions["q50"] <= predictions["q90"]).all()

    # Holdout reservation respected
    assert not metadata["final_holdout_evaluated"]
    assert metadata["max_target_day"] <= cfg["forecast"]["final_origin"]

    # Saved artefacts exist
    assert (tmp_path / "predictions.parquet").exists()
    assert (tmp_path / "scored_predictions.parquet").exists()
    assert (tmp_path / "metrics.csv").exists()
