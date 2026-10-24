from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from src.evaluation.backtesting import (BaseForecaster, DummyForecaster, construct_inputs,
    run_backtest, validate_predictions, attach_actuals, FUTURE_KEYS, KNOWN_CALENDAR)
from src.evaluation.splits import development_splits, final_holdout_spec
from src.data.load_data import STATIC


def test_exact_default_development_and_reserved_windows():
    from src.config import load_config
    cfg = load_config()
    folds = development_splits(cfg)
    assert [(f.origin, min(f.target_days), max(f.target_days)) for f in folds] == [
        (1801, 1802, 1829), (1829, 1830, 1857), (1857, 1858, 1885), (1885, 1886, 1913)]
    final = final_holdout_spec(cfg)
    assert list(final.target_days) == list(range(1914, 1942))


def test_split_refuses_final_holdout_as_development(tiny_panel):
    _, cfg = tiny_panel
    cfg["forecast"]["development_origins"] = [6, 12]
    with pytest.raises(ValueError, match="reserved final holdout"):
        development_splits(cfg)


def test_inputs_allowlist_and_expanding_history(tiny_panel):
    panel, cfg = tiny_panel
    for fold in development_splits(cfg):
        inputs = construct_inputs(panel, fold)
        assert inputs.history.day_num.max() == fold.origin
        assert len(inputs.history) == 3 * fold.origin
        assert set(inputs.future_known.columns) == set(FUTURE_KEYS + KNOWN_CALENDAR)
        assert set(inputs.static.columns) == {"series_id", *STATIC}
        assert set(inputs.future_known.target_day_num) == set(fold.target_days)
        assert "units_sold" not in inputs.future_known and "sell_price" not in inputs.future_known
        assert not any(c.startswith("snap") or "rolling" in c or "availability" in c for c in inputs.future_known)


def test_future_mutations_do_not_change_model_inputs(tiny_panel):
    panel, cfg = tiny_panel
    fold = development_splits(cfg)[0]
    original = construct_inputs(panel, fold)
    changed = panel.copy()
    mask = changed.day_num.gt(fold.origin)
    changed.loc[mask, "units_sold"] = 999999
    changed.loc[mask, "sell_price"] = 111
    changed.loc[mask, "snap"] = 1
    changed.loc[mask, "snap_CA"] = 1
    changed.loc[mask, "price_missing"] = True
    changed.loc[mask, "possible_prelaunch"] = True
    changed["future_rolling_mean"] = 12345  # Arbitrary accidental feature is excluded.
    again = construct_inputs(changed, fold)
    for name in ["history", "future_known", "static"]:
        pd.testing.assert_frame_equal(getattr(original, name), getattr(again, name))


class MeanSpyForecaster(DummyForecaster):
    """A train-mean spy only for checking fit boundaries, not a project baseline."""
    fitted_max_days = []

    def fit(self, inputs):
        assert inputs.history.day_num.max() <= inputs.origin
        self.fitted_max_days.append(int(inputs.history.day_num.max()))
        self.value = float(inputs.history.units_sold.mean())
        return super().fit(inputs)


def test_fold_fitted_preprocessing_cannot_use_validation_targets(tiny_panel):
    panel, cfg = tiny_panel
    one_fold = deepcopy(cfg)
    one_fold["forecast"]["development_origins"] = [6]
    original, _, _ = run_backtest(panel, one_fold, MeanSpyForecaster)
    changed = panel.copy()
    changed.loc[changed.day_num.gt(6), "units_sold"] = 1000000
    mutated, _, _ = run_backtest(changed, one_fold, MeanSpyForecaster)
    pd.testing.assert_frame_equal(original, mutated)
    assert MeanSpyForecaster.fitted_max_days[-2:] == [6, 6]


def test_dummy_runner_shape_schema_metrics_and_fresh_instances(tiny_panel, tmp_path):
    panel, cfg = tiny_panel
    models = []
    def factory():
        model = DummyForecaster()
        models.append(model)
        return model
    prediction, metrics, metadata = run_backtest(panel, cfg, factory, tmp_path)
    assert prediction.shape == (3 * 2 * 3, 10)
    assert len(models) == 2 and models[0] is not models[1]
    assert prediction.target_day_num.max() == 12 and not metadata["final_holdout_evaluated"]
    assert not prediction[["q10", "q50", "q90"]].notna().any().any()
    assert (metrics.wape == 1).all()
    assert metrics.pinball_q10.isna().all()
    assert metrics.n_rows.iloc[-1] == 18
    assert (tmp_path / "predictions.parquet").exists()


def test_final_holdout_mutations_never_change_development_results(tiny_panel):
    panel, cfg = tiny_panel
    prediction, metrics, _ = run_backtest(panel, cfg, DummyForecaster)
    changed = panel.copy()
    changed.loc[changed.day_num.gt(cfg["forecast"]["final_origin"]), "units_sold"] = np.nan
    again, again_metrics, _ = run_backtest(changed, cfg, DummyForecaster)
    pd.testing.assert_frame_equal(prediction, again)
    pd.testing.assert_frame_equal(metrics, again_metrics)


def test_factory_cannot_reuse_fitted_model(tiny_panel):
    panel, cfg = tiny_panel
    model = DummyForecaster()
    with pytest.raises(ValueError, match="fresh BaseForecaster"):
        run_backtest(panel, cfg, lambda: model)


@pytest.mark.parametrize("problem", ["length", "duplicate", "target_day", "date", "horizon", "nan_point", "negative", "crossing", "partial_quantile"])
def test_bad_forecasts_rejected(tiny_panel, problem):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    prediction = DummyForecaster().fit(inputs).predict(inputs)
    if problem == "length": prediction = prediction.iloc[:-1]
    if problem == "duplicate": prediction.iloc[0] = prediction.iloc[1]
    if problem == "target_day": prediction.loc[0, "target_day_num"] += 1
    if problem == "date": prediction.loc[0, "target_date"] += pd.Timedelta(days=1)
    if problem == "horizon": prediction.loc[0, "horizon"] = 99
    if problem == "nan_point": prediction.loc[0, "y_pred"] = np.nan
    if problem == "negative": prediction.loc[0, "y_pred"] = -1
    if problem == "crossing": prediction[["q10", "q50", "q90"]] = [3., 2., 1.]
    if problem == "partial_quantile": prediction.loc[0, "q10"] = 1
    with pytest.raises(ValueError): validate_predictions(prediction, inputs.future_known)


def test_valid_ordered_quantiles(tiny_panel):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    prediction = DummyForecaster().fit(inputs).predict(inputs)
    prediction[["q10", "q50", "q90"]] = [0., 1., 2.]
    validate_predictions(prediction, inputs.future_known)


def test_missing_actual_values_allowed_but_missing_keys_rejected(tiny_panel):
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    prediction = DummyForecaster().fit(inputs).predict(inputs)
    changed = panel.copy()
    mask = changed.day_num.eq(7)
    changed.loc[mask, "units_sold"] = np.nan
    assert attach_actuals(prediction, changed).actual.isna().sum() == 3
    with pytest.raises(ValueError, match="Unmatched evaluation"):
        attach_actuals(prediction, changed.loc[~mask])


def test_missing_training_rows_rejected(tiny_panel):
    panel, cfg = tiny_panel
    fold = development_splits(cfg)[0]
    with pytest.raises(ValueError, match="every indexed day"):
        construct_inputs(panel.iloc[1:], fold)


def test_future_calendar_inconsistency_rejected(tiny_panel):
    panel, cfg = tiny_panel
    changed = panel.copy()
    changed.loc[changed.day_num.eq(7), "date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="Future dates"):
        construct_inputs(changed, development_splits(cfg)[0])
