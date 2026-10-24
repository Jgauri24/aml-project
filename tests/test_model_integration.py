"""Cross-adapter integration checks, using only the existing synthetic fixtures."""
import gc
import weakref
from functools import partial

import numpy as np
import pandas as pd
import pytest

from examples.constant_forecaster import ExampleConstantForecaster
from src.evaluation.backtesting import (
    DummyForecaster, PREDICTION_COLUMNS, attach_actuals, construct_inputs,
    run_backtest, validate_predictions,
)
from src.evaluation.metrics import score_predictions
from src.evaluation.splits import development_splits


def test_example_factory_integration_and_saved_actuals(tiny_panel, tmp_path):
    panel, cfg = tiny_panel
    before = panel.copy(deep=True)
    predictions, metrics, metadata = run_backtest(
        panel, cfg, partial(ExampleConstantForecaster, value=2.0), tmp_path)
    assert list(predictions) == PREDICTION_COLUMNS
    assert predictions.shape == (18, 10)
    assert predictions.y_pred.eq(2.0).all()
    assert predictions[["q10", "q50", "q90"]].isna().all().all()
    assert "actual" not in predictions
    assert len(metrics) == 3 and metrics.scope.iloc[-1] == "all_folds_pooled"
    assert metrics.n_observed.iloc[-1] == 18
    assert metadata["max_target_day"] == 12 and not metadata["final_holdout_evaluated"]
    scored = pd.read_parquet(tmp_path / "scored_predictions.parquet")
    assert "actual" in scored and len(scored) == len(predictions)
    pd.testing.assert_frame_equal(panel, before)


def test_example_requires_fit_for_the_requested_origin(tiny_panel):
    panel, cfg = tiny_panel
    first, second = [construct_inputs(panel, f) for f in development_splits(cfg)]
    model = ExampleConstantForecaster()
    with pytest.raises(RuntimeError, match="Fit a fresh"):
        model.predict(first)
    model.fit(first)
    with pytest.raises(RuntimeError, match="Fit a fresh"):
        model.predict(second)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_example_rejects_invalid_constant(value):
    with pytest.raises(ValueError):
        ExampleConstantForecaster(value)


def test_shuffled_predictions_align_actuals_by_keys(tiny_panel):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    prediction = ExampleConstantForecaster().fit(inputs).predict(inputs)
    # Distinct predictions per row make a positional alignment error observable.
    prediction["y_pred"] = np.arange(len(prediction), dtype=float)
    shuffled = prediction.sample(frac=1, random_state=42).reset_index(drop=True)
    validate_predictions(shuffled, inputs.future_known)
    left = attach_actuals(prediction, panel).sort_values(["series_id", "horizon"]).reset_index(drop=True)
    right = attach_actuals(shuffled, panel).sort_values(["series_id", "horizon"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    # Explicitly compare finite metrics; unavailable pinball is intentionally NaN.
    for metric in ["mae", "rmse", "wape"]:
        assert score_predictions(left)[metric] == score_predictions(right)[metric]


class QuantileFixtureForecaster(DummyForecaster):
    """Test fixture for evaluator quantile plumbing; never an experimental model."""
    def predict(self, inputs):
        prediction = super().predict(inputs)
        prediction[["q10", "q50", "q90"]] = [0.0, 1.0, 2.0]
        return prediction


def test_quantiles_score_through_engine(tiny_panel):
    panel, cfg = tiny_panel
    predictions, metrics, _ = run_backtest(panel, cfg, QuantileFixtureForecaster)
    actuals = attach_actuals(predictions, panel).actual.to_numpy(dtype=float)
    pooled = metrics.loc[metrics.scope.eq("all_folds_pooled")].iloc[0]
    for name, q, value in [("q10", 0.1, 0), ("q50", 0.5, 1), ("q90", 0.9, 2)]:
        residual = actuals - value
        expected = np.maximum(q * residual, (q - 1) * residual).mean()
        assert pooled[f"pinball_{name}"] == pytest.approx(expected)


def test_nullable_missing_point_forecasts_rejected(tiny_panel):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    prediction = DummyForecaster().fit(inputs).predict(inputs)
    prediction["y_pred"] = pd.Series(pd.NA, index=prediction.index, dtype="Float64")
    with pytest.raises(ValueError, match="finite"):
        validate_predictions(prediction, inputs.future_known)


@pytest.mark.parametrize("problem", ["missing_column", "unknown_series", "mixed_model", "nonfinite_quantile", "negative_quantile"])
def test_adapter_output_contract_rejects_invalid_fields(tiny_panel, problem):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    prediction = DummyForecaster().fit(inputs).predict(inputs)
    if problem == "missing_column":
        prediction = prediction.drop(columns="q50")
    elif problem == "unknown_series":
        prediction["series_id"] = prediction.series_id.astype(str)
        prediction.loc[0, "series_id"] = "not_a_requested_series"
    elif problem == "mixed_model":
        prediction.loc[0, "model"] = "another_model"
    elif problem == "nonfinite_quantile":
        prediction[["q10", "q50", "q90"]] = [0.0, 1.0, np.inf]
    else:
        prediction[["q10", "q50", "q90"]] = [-1.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        validate_predictions(prediction, inputs.future_known)


def test_factory_must_return_forecaster(tiny_panel):
    panel, cfg = tiny_panel
    with pytest.raises(ValueError, match="fresh BaseForecaster"):
        run_backtest(panel, cfg, lambda: object())


def test_model_name_cannot_change_across_folds(tiny_panel):
    class ChangingName(DummyForecaster):
        def predict(self, inputs):
            prediction = super().predict(inputs)
            prediction["model"] = f"model_{inputs.origin}"
            return prediction
    panel, cfg = tiny_panel
    with pytest.raises(ValueError, match="Model name changed"):
        run_backtest(panel, cfg, ChangingName)


def test_previous_fold_models_are_released(tiny_panel):
    """Holding all fitted objects retains four deep-model weight sets unnecessarily."""
    panel, cfg = tiny_panel
    references = []
    def factory():
        gc.collect()
        assert all(ref() is None for ref in references), "Previous fitted model retained by engine"
        model = DummyForecaster()
        references.append(weakref.ref(model))
        return model
    run_backtest(panel, cfg, factory)
