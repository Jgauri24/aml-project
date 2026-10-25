"""Tests for the M9 LightGBM forecaster.

Uses the ``tiny_panel`` / ``tiny_project`` fixtures from conftest.py to
exercise the full fit→predict pipeline without requiring real M5 data.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm", reason="Run the complete model suite with .venv-lightgbm/bin/python")

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


def test_saved_model_reloads_without_history_or_refitting(tiny_inputs, tmp_path):
    from copy import deepcopy
    import pandas as pd
    from src.evaluation.backtesting import validate_predictions
    model = LightGBMForecaster({'n_estimators':5,'n_jobs':1,'min_child_samples':1}).fit(tiny_inputs)
    expected = model.predict(tiny_inputs)
    model.save(tmp_path)
    assert (tmp_path/'model.txt').is_file() and (tmp_path/'state.json').is_file()
    restored = LightGBMForecaster.load(tmp_path)
    inference = deepcopy(tiny_inputs)
    inference.history = inference.history.iloc[:0]
    actual = restored.predict(inference)
    pd.testing.assert_frame_equal(actual,expected,check_exact=False,rtol=1e-12,atol=1e-12)
    validate_predictions(actual,inference.future_known,'lightgbm')


def test_saved_booster_and_feature_state_cannot_be_mixed(tiny_inputs,tmp_path):
    model=LightGBMForecaster({'n_estimators':2,'n_jobs':1}).fit(tiny_inputs)
    model.save(tmp_path)
    with (tmp_path/'model.txt').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError,match='does not match'):
        LightGBMForecaster.load(tmp_path)


def test_runner_persists_tuning_metadata_models_and_all_fold_states(tiny_panel,tmp_path):
    import json
    import pandas as pd
    from src.models.lightgbm_forecaster import run_lightgbm_forecaster
    panel,cfg=tiny_panel
    grid=[{'num_leaves':2,'n_estimators':2},{'num_leaves':3,'n_estimators':3}]
    predictions,metrics,metadata=run_lightgbm_forecaster(
        panel,cfg,str(tmp_path/'run'),tune=True,params={'n_jobs':1},tuning_grid=grid)
    destination=tmp_path/'run'
    saved=json.loads((destination/'execution_metadata.json').read_text())
    assert saved==metadata
    assert saved['tuning_origins']==[6]
    assert saved['selected_candidate'] in grid
    assert saved['wall_seconds'] >= saved['tuning_seconds'] + saved['backtest_seconds']
    tuning=json.loads((destination/'tuning_results.json').read_text())
    assert len(tuning)==2
    assert all([fold['origin'] for fold in row['fold_metrics']]==[6] for row in tuning)
    assert len(saved['model_artifacts'])==2
    for artifact,fold in zip(saved['model_artifacts'],development_splits(cfg)):
        inputs=construct_inputs(panel,fold)
        loaded=LightGBMForecaster.load(destination/artifact['directory'])
        expected=predictions.loc[predictions.origin.eq(fold.origin)].reset_index(drop=True)
        pd.testing.assert_frame_equal(loaded.predict(inputs).reset_index(drop=True),expected,
                                      check_exact=False,rtol=1e-12,atol=1e-12)
    assert metadata['final_holdout_evaluated'] is False
    assert predictions.target_day_num.max()==12
    assert (destination/'fit_summary.csv').is_file()


def test_tuning_rejects_unavailable_wape(tiny_panel):
    panel,cfg=tiny_panel
    panel=panel.copy()
    panel['units_sold']=0.0
    with pytest.raises(ValueError,match='finite tuning WAPE'):
        tune_hyperparameters(panel,cfg,grid=[{'num_leaves':2,'n_estimators':2,'n_jobs':1}])


def test_model_cannot_predict_beyond_trained_horizon(tiny_inputs):
    from dataclasses import replace
    model=LightGBMForecaster({'n_estimators':2,'n_jobs':1}).fit(tiny_inputs)
    with pytest.raises(ValueError,match='exceeds the trained horizon'):
        model.predict(replace(tiny_inputs,horizon=tiny_inputs.horizon+1))


def test_direct_model_rejects_missing_training_horizons(tiny_inputs):
    from dataclasses import replace
    short=replace(tiny_inputs,history=tiny_inputs.history.loc[tiny_inputs.history.day_num.le(2)].copy(),origin=2)
    with pytest.raises(ValueError,match='every configured horizon'):
        LightGBMForecaster({'n_estimators':2,'n_jobs':1}).fit(short)
