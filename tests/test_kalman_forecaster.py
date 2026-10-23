"""Synthetic unit tests for the M7 state-space mathematical core."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.models.kalman_forecaster import (
    VARIANTS, KalmanForecaster, build_system, forecast, initial_state, kalman_filter,
    make_kalman_factory, model_name,
)
from src.evaluation.backtesting import PREDICTION_COLUMNS, construct_inputs, run_backtest
from src.evaluation.splits import development_splits
from src.models.kalman_single_sku import kalman_filter as m6_kalman_filter


def _variances(variant):
    values = {"level": 0.2, "observation": 0.7}
    if "trend" in variant:
        values["slope"] = 0.05
    if "seasonal" in variant:
        values["seasonal"] = 0.01
    return values


def test_registry_and_unknown_variant():
    assert VARIANTS == ("local_level", "local_level_seasonal", "local_trend", "local_trend_seasonal")
    assert model_name("local_trend") == "kalman_local_trend"
    with pytest.raises(ValueError, match="Unknown Kalman variant"):
        build_system("not_a_variant", {"level": 1, "observation": 1})


@pytest.mark.parametrize("variant, dimension", [("local_level", 1), ("local_level_seasonal", 7), ("local_trend", 2), ("local_trend_seasonal", 8)])
def test_system_dimensions_and_entries(variant, dimension):
    T, Z, Q, R = build_system(variant, _variances(variant))
    assert T.shape == Q.shape == (dimension, dimension)
    assert Z.shape == (dimension,)
    assert R == 0.7 and Q[0, 0] == 0.2 and Z[0] == 1.0
    if "trend" in variant:
        assert T[0, 1] == 1.0 and T[1, 1] == 0.98 and Z[1] == 0.0
    if "seasonal" in variant:
        offset = 2 if "trend" in variant else 1
        assert Z[offset] == 1.0
        assert np.allclose(T[offset:offset + 5, offset + 1:offset + 6], np.eye(5))
        assert np.allclose(T[offset + 5, offset:offset + 6], -1.0)


def test_generic_local_level_likelihood_matches_m6():
    y = np.array([2.0, np.nan, 3.0, 1.0, 4.0])
    Q_value, R, mu0, P0 = 0.3, 0.8, 1.2, 2.5
    T, Z, Q, _ = build_system("local_level", {"level": Q_value, "observation": R})
    generic = kalman_filter(y, T, Z, Q, R, np.array([mu0]), np.array([[P0]]))
    m6 = m6_kalman_filter(y, Q=Q_value, R=R, mu0=mu0, P0=P0)
    difference = abs(generic.log_lik - m6.log_lik)
    assert generic.log_lik == pytest.approx(m6.log_lik, abs=1e-12), (
        f"M6 likelihood={m6.log_lik}, generic likelihood={generic.log_lik}, "
        f"absolute difference={difference}, tolerance=1e-12"
    )


def test_trend_hand_calculation_and_forecast():
    T, Z, Q, R = build_system("local_trend", {"level": 0.0, "slope": 0.0, "observation": 1.0}, damping=1.0)
    result = kalman_filter(np.array([3.0]), T, Z, Q, R, np.array([1.0, 2.0]), np.eye(2))
    # Prediction is [3, 2], P=[[2,1],[1,1]], F=3, K=[2/3,1/3].
    assert np.allclose(result.states[0], [3.0, 2.0])
    assert result.innovation_variances[0] == pytest.approx(3.0)
    fc = forecast(result.states[-1], result.covariances[-1], T, Z, Q, R, horizon=2)
    assert np.allclose(fc["mean"], [5.0, 7.0])


def test_missing_skips_update_but_propagates():
    T, Z, Q, R = build_system("local_trend", {"level": 0.2, "slope": 0.1, "observation": 1.0}, damping=1.0)
    result = kalman_filter(np.array([1.0, np.nan]), T, Z, Q, R, np.array([0.0, 1.0]), np.eye(2))
    assert np.isnan(result.innovations[1])
    assert np.allclose(result.states[1], T @ result.states[0])
    assert result.covariances[1, 0, 0] > result.covariances[0, 0, 0]


def test_forecast_variance_grows_over_gap_when_level_noise_positive():
    T, Z, Q, R = build_system("local_level", {"level": 0.5, "observation": 1.0})
    result = kalman_filter(np.array([1.0, np.nan, np.nan]), T, Z, Q, R, np.array([0.0]), np.array([[1.0]]))
    assert result.covariances[2, 0, 0] > result.covariances[1, 0, 0]
    fc = forecast(result.states[-1], result.covariances[-1], T, Z, Q, R, 4)
    assert np.all(np.diff(fc["variance"]) > 0)


def test_interval_width_is_consistent_with_forecast_variance():
    T, Z, Q, R = build_system("local_level", {"level": 0.5, "observation": 1.0})
    fc = forecast(np.array([10.0]), np.array([[1.0]]), T, Z, Q, R, 4)
    # Mean is safely above zero, so clipping does not change the Gaussian interval width.
    expected = (norm.ppf(0.9) - norm.ppf(0.1)) * np.sqrt(fc["variance"])
    assert np.allclose(fc["q90"] - fc["q10"], expected)


def test_seasonal_phase_forecasts_correct_weekday_pattern_for_28_days():
    pattern = np.array([1., 2., 3., 4., 5., 6., 7.])
    weekdays = np.arange(35) % 7
    y = pattern[weekdays]
    T, Z, Q, R = build_system("local_level_seasonal", {"level": 0.0, "seasonal": 0.0, "observation": 1e-8})
    prior = initial_state(y, "local_level_seasonal", weekdays)
    # Start from the known deterministic phase so this tests transition/forecast alignment.
    result = kalman_filter(y, T, Z, Q, R, prior.state, prior.covariance)
    fc = forecast(result.states[-1], result.covariances[-1], T, Z, Q, R, 28)
    expected = pattern[(weekdays[-1] + np.arange(1, 29)) % 7]
    assert np.allclose(fc["mean"], expected, atol=2e-4), "weekly phase shifted or otherwise misaligned"


def test_constructor_validation():
    with pytest.raises(ValueError, match="fit_window"):
        KalmanForecaster("local_level", fit_window=0)
    with pytest.raises(ValueError, match="stable range"):
        KalmanForecaster("local_level", damping=1.1)
    with pytest.raises(ValueError, match="n_jobs"):
        KalmanForecaster("local_level", n_jobs=0)
    assert KalmanForecaster("local_level", fit_window=None, n_jobs=-1).fit_window is None


def test_scaled_log_likelihood_correction_applied_once(tiny_panel):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    model = KalmanForecaster("local_level", fit_window=None).fit(inputs)
    row = model.fit_summary_.iloc[0]
    state = model._states[row.series_id]
    frame = inputs.history.loc[inputs.history.series_id.astype(str).eq(row.series_id)].sort_values("day_num")
    y = frame.units_sold.to_numpy(dtype=float) / row.scale
    prior = initial_state(y, "local_level")
    scaled_ll = kalman_filter(y, state.T, state.Z, state.Q, state.R, prior.state, prior.covariance).log_lik
    assert row.log_likelihood == pytest.approx(scaled_ll - row.observed_targets * np.log(row.scale))


def test_adapter_schema_quantiles_and_summary(tiny_panel):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    model = KalmanForecaster("local_trend_seasonal", fit_window=None).fit(inputs)
    prediction = model.predict(inputs)
    assert list(prediction) == PREDICTION_COLUMNS
    assert len(prediction) == len(inputs.future_known)
    assert not prediction.duplicated(["series_id", "origin", "horizon"]).any()
    assert np.isfinite(prediction[["y_pred", "q10", "q50", "q90"]].to_numpy(dtype=float)).all()
    assert prediction.y_pred.ge(0).all() and prediction.q10.ge(0).all()
    assert prediction.y_pred.equals(prediction.q50)
    assert (prediction.q10 <= prediction.q50).all() and (prediction.q50 <= prediction.q90).all()
    assert (prediction.q90 - prediction.q10 >= 0).all()
    required = {"series_id", "variant", "fit_window_start", "fit_window_end", "calendar_days_in_window",
                "observed_targets", "level_variance", "trend_variance", "seasonal_variance",
                "observation_variance", "log_likelihood", "aic", "convergence_status",
                "optimizer_message", "objective_value", "fallback", "fallback_reason"}
    assert not (required - set(model.fit_summary_))
    assert model.fit_summary_.series_id.is_monotonic_increasing


def test_all_nan_and_zero_windows_fallback_valid_output(tiny_panel):
    panel, cfg = tiny_panel
    panel = panel.copy()
    series = list(panel.series_id.cat.categories)
    panel.loc[panel.series_id == series[0], "units_sold"] = np.nan
    panel.loc[panel.series_id == series[1], "units_sold"] = 0.0
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    with pytest.warns(RuntimeWarning):
        model = KalmanForecaster("local_level", fit_window=None).fit(inputs)
    prediction = model.predict(inputs)
    assert np.isfinite(prediction[["y_pred", "q10", "q50", "q90"]].to_numpy(dtype=float)).all()
    assert (prediction.q10 <= prediction.q50).all() and (prediction.q50 <= prediction.q90).all()
    reasons = set(model.fit_summary_.fallback_reason.dropna())
    assert {"all_nan_window", "all_zero_window"} <= reasons


def test_optimizer_failure_is_explicit_fallback(tiny_panel, monkeypatch):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    def fail(*args, **kwargs):
        raise RuntimeError("forced optimiser failure")
    monkeypatch.setattr("src.models.kalman_forecaster.minimize", fail)
    with pytest.warns(RuntimeWarning, match="optimiser"):
        model = KalmanForecaster("local_level").fit(inputs)
    assert model.fit_summary_.fallback.all()
    nonzero = model.fit_summary_.query("fallback_reason == 'optimizer_failure'")
    assert not nonzero.empty
    assert set(model.fit_summary_.convergence_status) == {"fallback"}


def test_aic_counts_only_estimated_training_parameters(tiny_panel):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    model = KalmanForecaster("local_trend", fit_window=None).fit(inputs)
    rows = model.fit_summary_.loc[~model.fit_summary_.fallback]
    assert not rows.empty
    assert (rows.estimated_parameter_count == 3).all()  # level, slope, observation; damping is fixed.
    assert np.allclose(rows.aic, 2 * rows.estimated_parameter_count - 2 * rows.log_likelihood)


def test_predict_guards_and_factory(tiny_panel):
    panel, cfg = tiny_panel
    folds = development_splits(cfg)
    first, second = construct_inputs(panel, folds[0]), construct_inputs(panel, folds[1])
    with pytest.raises(RuntimeError, match="fit"):
        KalmanForecaster("local_level").predict(first)
    model = make_kalman_factory("local_level")()
    model.fit(first)
    with pytest.raises(RuntimeError, match="origin"):
        model.predict(second)
    assert make_kalman_factory("local_level")() is not make_kalman_factory("local_level")()


def test_post_origin_sales_and_prices_do_not_change_predictions(tiny_panel):
    panel, cfg = tiny_panel
    fold = development_splits(cfg)[0]
    corrupt = panel.copy()
    corrupt.loc[corrupt.day_num > fold.origin, ["units_sold", "sell_price"]] = 999.0
    clean_inputs, corrupt_inputs = construct_inputs(panel, fold), construct_inputs(corrupt, fold)
    clean = KalmanForecaster("local_level").fit(clean_inputs).predict(clean_inputs)
    changed = KalmanForecaster("local_level").fit(corrupt_inputs).predict(corrupt_inputs)
    np.testing.assert_allclose(clean[["y_pred", "q10", "q50", "q90"]], changed[["y_pred", "q10", "q50", "q90"]])


def test_parallel_equals_serial(tiny_panel):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    serial = KalmanForecaster("local_level", n_jobs=1).fit(inputs)
    parallel = KalmanForecaster("local_level", n_jobs=2).fit(inputs)
    np.testing.assert_allclose(serial.predict(inputs)[["y_pred", "q10", "q50", "q90"]],
                               parallel.predict(inputs)[["y_pred", "q10", "q50", "q90"]], atol=1e-10)
    pd.testing.assert_frame_equal(serial.fit_summary_, parallel.fit_summary_, check_exact=False, atol=1e-10)


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_variants_run_backtest_on_tiny_panel(tiny_panel, variant):
    panel, cfg = tiny_panel
    predictions, _, metadata = run_backtest(panel, cfg, make_kalman_factory(variant, fit_window=None))
    assert list(predictions) == PREDICTION_COLUMNS
    assert len(predictions) == 18 and metadata["max_target_day"] <= cfg["forecast"]["final_origin"]
