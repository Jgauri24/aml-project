"""Regression checks for the repaired M8/M9 forecasting contracts."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtesting import construct_inputs
from src.evaluation.splits import development_splits
from src.features.feature_engineering import FeatureEngineer, _encode_calendar


def test_training_features_do_not_use_label_day_price(tiny_panel):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    changed = deepcopy(inputs)
    sid = inputs.history.series_id.iloc[0]
    target_day = inputs.origin
    changed.history.loc[changed.history.series_id.eq(sid) & changed.history.day_num.eq(target_day),
                        "sell_price"] = 9999.0
    before, _ = FeatureEngineer().fit(inputs).build_train_features(inputs)
    after, _ = FeatureEngineer().fit(changed).build_train_features(changed)
    # This is the final label day, so there are no later training examples that
    # could legitimately use its newly observed price as historical information.
    pd.testing.assert_frame_equal(before, after)


def test_missing_events_are_encoded_as_absent():
    calendar = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=4),
                             "event_name_1": [None, pd.NA, np.nan, ""]})
    assert _encode_calendar(calendar).has_event_name_1.tolist() == [0, 0, 0, 0]


def test_multi_horizon_model_trains_on_observed_horizons(tiny_panel, monkeypatch):
    lgb = pytest.importorskip("lightgbm")
    from src.models.lightgbm_forecaster import LightGBMForecaster

    original_fit = lgb.LGBMRegressor.fit
    observed = {}

    def capture_fit(self, X, y, **kwargs):
        observed["horizon"] = X["horizon"].copy()
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(lgb.LGBMRegressor, "fit", capture_fit)
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    LightGBMForecaster(params={"n_estimators": 2, "n_jobs": 1}).fit(inputs)
    assert observed["horizon"].notna().all(), "All training horizons are unavailable"
    assert observed["horizon"].nunique() > 1, "Training must represent multiple horizons"
