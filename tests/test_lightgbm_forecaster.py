"""Tests for the M9 LightGBM forecaster.

Uses the ``tiny_panel`` / ``tiny_project`` fixtures from conftest.py to
exercise the full fit→predict pipeline without requiring real M5 data.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocess import run_preprocessing
from src.evaluation.backtesting import (
    PREDICTION_COLUMNS, construct_inputs, run_backtest,
)
from src.evaluation.splits import development_splits
from src.models.lightgbm_forecaster import LightGBMForecaster, tune_hyperparameters


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_inputs(tiny_panel):
    """Return ForecastInputs for the first development fold of the tiny dataset."""
    panel, cfg = tiny_panel
    fold = development_splits(cfg)[0]
    return construct_inputs(panel, fold)


@pytest.fixture
def tiny_panel_preprocessed(tiny_project):
    """Return a panel built through run_preprocessing for the tiny project."""
    cfg, _ = tiny_project
    panel, _ = run_preprocessing(cfg)
    return panel, cfg


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestLightGBMForecasterUnit:
    def test_fit_returns_self(self, tiny_inputs):
        model = LightGBMForecaster()
        result = model.fit(tiny_inputs)
        assert result is model

    def test_predict_returns_prediction_columns(self, tiny_inputs):
        model = LightGBMForecaster()
        model.fit(tiny_inputs)
        pred = model.predict(tiny_inputs)
        assert list(pred.columns) == PREDICTION_COLUMNS

    def test_predictions_nonneg_finite(self, tiny_inputs):
        model = LightGBMForecaster()
        model.fit(tiny_inputs)
        pred = model.predict(tiny_inputs)
        assert pred["y_pred"].ge(0).all()
        assert np.isfinite(pred["y_pred"].to_numpy()).all()

    def test_model_name_is_lightgbm(self, tiny_inputs):
        model = LightGBMForecaster()
        model.fit(tiny_inputs)
        pred = model.predict(tiny_inputs)
        assert pred["model"].eq("lightgbm").all()

    def test_prediction_row_count(self, tiny_inputs):
        """Each series × each horizon should have exactly one row."""
        model = LightGBMForecaster()
        model.fit(tiny_inputs)
        pred = model.predict(tiny_inputs)
        n_series = tiny_inputs.history["series_id"].nunique()
        assert len(pred) == n_series * tiny_inputs.horizon

    def test_predict_before_fit_raises(self, tiny_inputs):
        model = LightGBMForecaster()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(tiny_inputs)

    def test_feature_importance_set_after_fit(self, tiny_inputs):
        model = LightGBMForecaster()
        model.fit(tiny_inputs)
        assert model.feature_importance_ is not None
        assert "feature" in model.feature_importance_.columns
        assert "importance_gain" in model.feature_importance_.columns
        assert len(model.feature_importance_) > 0

    def test_fit_summary_set_after_fit(self, tiny_inputs):
        model = LightGBMForecaster()
        model.fit(tiny_inputs)
        assert model.fit_summary_ is not None
        assert "origin" in model.fit_summary_.columns

    def test_quantiles_are_nan(self, tiny_inputs):
        """Quantiles should be NaN (point-only forecaster)."""
        model = LightGBMForecaster()
        model.fit(tiny_inputs)
        pred = model.predict(tiny_inputs)
        assert pred[["q10", "q50", "q90"]].isna().all().all()

    def test_wrong_origin_raises(self, tiny_panel):
        panel, cfg = tiny_panel
        folds = development_splits(cfg)
        inputs_0 = construct_inputs(panel, folds[0])
        inputs_1 = construct_inputs(panel, folds[1])
        model = LightGBMForecaster()
        model.fit(inputs_0)
        with pytest.raises(RuntimeError):
            model.predict(inputs_1)


# ── Integration: run_backtest compatibility ───────────────────────────────────

class TestLightGBMBacktest:
    def test_run_backtest_completes(self, tiny_panel_preprocessed):
        """LightGBM must produce valid output through run_backtest."""
        panel, cfg = tiny_panel_preprocessed
        predictions, metrics, metadata = run_backtest(
            panel, cfg,
            factory=lambda: LightGBMForecaster(),
        )
        n_series = panel["series_id"].nunique()
        n_folds = len(development_splits(cfg))
        horizon = cfg["forecast"]["horizon"]

        assert len(predictions) == n_series * n_folds * horizon
        assert list(predictions.columns) == PREDICTION_COLUMNS
        assert predictions["y_pred"].ge(0).all()
        assert np.isfinite(predictions["y_pred"].to_numpy()).all()
        # Verify fold-level metrics exist
        fold_metrics = metrics.loc[metrics["scope"] == "fold_pooled"]
        assert len(fold_metrics) == n_folds

    def test_factory_creates_fresh_models(self, tiny_panel_preprocessed):
        """run_backtest should reject reused model instances."""
        panel, cfg = tiny_panel_preprocessed
        # run_backtest already enforces this; we just confirm no exception from our factory
        predictions, _, _ = run_backtest(
            panel, cfg,
            factory=lambda: LightGBMForecaster(),
        )
        assert len(predictions) > 0


# ── Tuning ────────────────────────────────────────────────────────────────────

class TestTuning:
    def test_tune_returns_dict(self, tiny_panel_preprocessed):
        panel, cfg = tiny_panel_preprocessed
        grid = [
            {"num_leaves": 15, "learning_rate": 0.1, "n_estimators": 20},
            {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 20},
        ]
        result = tune_hyperparameters(panel, cfg, grid=grid)
        assert isinstance(result, dict)
        # Best config must be one of our candidates
        assert result in grid or result == {}
