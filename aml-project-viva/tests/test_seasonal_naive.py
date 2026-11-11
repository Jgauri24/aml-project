"""M5 mathematical, missingness, leakage and shared-engine checks; no M5 CSVs needed."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from scripts.run_seasonal_naive import comparison_conclusion, run_experiment, validate_integrity
from src.data.preprocess import run_preprocessing
from src.evaluation.backtesting import (
    ForecastInputs, FUTURE_KEYS, PREDICTION_COLUMNS, construct_inputs,
    run_backtest, validate_predictions,
)
from src.evaluation.metrics import score_predictions
from src.evaluation.splits import development_splits
from src.models.seasonal_naive import VARIANTS, SeasonalNaiveForecaster, make_seasonal_naive_factory


def make_inputs(values: dict[str, list[float]], horizon: int = 28) -> ForecastInputs:
    """Minimal independent calendar grid; differing weekday values reveal phase errors."""
    origin = max(len(series) for series in values.values())
    history = pd.DataFrame([
        {"series_id": sid, "day_num": day, "units_sold": value}
        for sid, series in values.items() for day, value in enumerate(series, start=1)
    ])
    grid = pd.DataFrame([
        {"series_id": sid, "origin": origin, "horizon": h, "target_day_num": origin + h,
         "target_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=origin + h - 1)}
        for sid in values for h in range(1, horizon + 1)
    ])
    return ForecastInputs(history, pd.DataFrame(), grid, origin, horizon)


def forecast(inputs: ForecastInputs, variant: str) -> pd.DataFrame:
    return SeasonalNaiveForecaster(variant).fit(inputs).predict(inputs)


@pytest.mark.parametrize("horizon", [1, 7, 10, 28, 35])
def test_last_value_and_exact_weekly_alignment(horizon):
    pattern = [5, 8, 4, 10, 7, 6, 9]
    inputs = make_inputs({"A": pattern * 2}, horizon)
    assert forecast(inputs, "last_value").y_pred.eq(9).all()
    np.testing.assert_array_equal(forecast(inputs, "weekly_seasonal").y_pred,
                                  np.resize(pattern, horizon))


def test_weekly_pattern_is_last_week_not_historical_weekday_mean():
    inputs = make_inputs({"A": [100] * 7 + [1, 2, 3, 4, 5, 6, 7]})
    np.testing.assert_array_equal(forecast(inputs, "weekly_seasonal").y_pred,
                                  np.tile(np.arange(1, 8), 4))


@pytest.mark.parametrize("variant", VARIANTS)
def test_genuine_zeros_are_valid(variant):
    inputs = make_inputs({"A": [0] * 14})
    model = SeasonalNaiveForecaster(variant).fit(inputs)
    assert model.predict(inputs).y_pred.eq(0).all()
    assert model.fit_summary_.weekly_last_week.eq(7).all()
    assert model.fit_summary_.last_value_source_day.eq(14).all()


def test_last_value_searches_earlier_finite_values():
    inputs = make_inputs({"A": [4, 0, np.nan, np.inf, -np.inf]})
    model = SeasonalNaiveForecaster("last_value").fit(inputs)
    assert model.predict(inputs).y_pred.eq(0).all()
    assert model.fit_summary_.last_value_source_day.iloc[0] == 2
    assert model.fit_summary_.last_value_fallback.iloc[0] == "earlier_observation"


def test_weekday_search_then_latest_overall_fallback_and_diagnostics():
    values = list(np.tile(np.arange(1, 8, dtype=float), 3))
    for day in [8, 15, 2, 9, 16]:
        values[day - 1] = np.nan
    values[17 - 1] = 0
    inputs = make_inputs({"A": values})
    model = SeasonalNaiveForecaster().fit(inputs)
    np.testing.assert_array_equal(model.predict(inputs).y_pred, np.tile([1, 7, 0, 4, 5, 6, 7], 4))
    summary = model.fit_summary_.iloc[0]
    assert summary.weekly_earlier_same_weekday == 1
    assert summary.weekly_latest_overall == 1
    assert summary.weekly_last_week == 5
    assert json.loads(summary.weekly_source_days) == [1, 21, 17, 18, 19, 20, 21]


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_missing_falls_back_to_zero(variant):
    inputs = make_inputs({"A": [np.nan] * 14})
    model = SeasonalNaiveForecaster(variant).fit(inputs)
    assert model.predict(inputs).y_pred.eq(0).all()
    assert model.fit_summary_.last_value_fallback.iloc[0] == "all_missing_zero"
    assert model.fit_summary_.weekly_all_missing_zero.iloc[0] == 7


def test_short_history_and_missing_calendar_rows_keep_weekday_positions():
    inputs = make_inputs({"A": [1, 2, 3]})
    np.testing.assert_array_equal(forecast(inputs, "weekly_seasonal").y_pred,
                                  np.tile([3, 3, 3, 3, 1, 2, 3], 4))
    inputs = make_inputs({"A": list(range(1, 15))})
    inputs.history = inputs.history.loc[~inputs.history.day_num.isin([8, 11])]
    np.testing.assert_array_equal(forecast(inputs, "weekly_seasonal").y_pred,
                                  np.tile([1, 9, 10, 4, 12, 13, 14], 4))


@pytest.mark.parametrize("variant", VARIANTS)
def test_multiple_series_schema_exact_keys_no_mutation_and_determinism(variant):
    inputs = make_inputs({"A": [1, 2, 3, 4, 5, 6, 7], "B": [10, 20, 30, 40, 50, 60, 0]})
    inputs.history = inputs.history.sample(frac=1, random_state=7)
    inputs.future_known = inputs.future_known.sample(frac=1, random_state=8)
    before = deepcopy(inputs)
    first = forecast(inputs, variant)
    second = forecast(inputs, variant)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first[FUTURE_KEYS], inputs.future_known[FUTURE_KEYS])
    assert list(first) == PREDICTION_COLUMNS
    assert first[["q10", "q50", "q90"]].isna().all().all()
    assert first.model.eq(f"naive_{variant}").all()
    assert "actual" not in first
    assert np.isfinite(first.y_pred).all() and first.y_pred.ge(0).all()
    a = first.loc[first.series_id.eq("A")].sort_values("horizon").y_pred
    b = first.loc[first.series_id.eq("B")].sort_values("horizon").y_pred
    np.testing.assert_array_equal(a, [7] * 28 if variant == "last_value" else np.tile(np.arange(1, 8), 4))
    np.testing.assert_array_equal(b, [0] * 28 if variant == "last_value" else np.tile([10, 20, 30, 40, 50, 60, 0], 4))
    for name in ["history", "static", "future_known"]:
        pd.testing.assert_frame_equal(getattr(inputs, name), getattr(before, name))


def test_nullable_missing_values():
    inputs = make_inputs({"A": [1, np.nan, 0]})
    inputs.history["units_sold"] = inputs.history.units_sold.astype("Float64")
    assert forecast(inputs, "last_value").y_pred.eq(0).all()
    assert np.isfinite(forecast(inputs, "weekly_seasonal").y_pred).all()


def test_before_fit_wrong_origin_and_missing_series_rejected():
    inputs = make_inputs({"A": [1, 2, 3]})
    model = SeasonalNaiveForecaster()
    with pytest.raises(RuntimeError, match="Fit this origin"):
        model.predict(inputs)
    model.fit(inputs)
    with pytest.raises(RuntimeError, match="Fit this origin"):
        model.predict(replace(inputs, origin=4))
    unknown = deepcopy(inputs)
    unknown.future_known["series_id"] = "unknown"
    with pytest.raises(ValueError, match="missing from fitted history"):
        model.predict(unknown)
    inputs.future_known["origin"] = 4
    with pytest.raises(ValueError, match="fitted origin"):
        model.predict(inputs)


@pytest.mark.parametrize("problem", ["future", "negative", "duplicate", "fractional_day", "empty"])
def test_invalid_training_history_rejected(problem):
    inputs = make_inputs({"A": [1, 2, 3]})
    if problem == "future":
        inputs.history.loc[0, "day_num"] = 4
    elif problem == "negative":
        inputs.history.loc[0, "units_sold"] = -1
    elif problem == "duplicate":
        inputs.history = pd.concat([inputs.history, inputs.history.iloc[:1]])
    elif problem == "fractional_day":
        inputs.history["day_num"] = inputs.history.day_num.astype(float)
        inputs.history.loc[0, "day_num"] = 1.5
    else:
        inputs.history = inputs.history.iloc[:0]
    with pytest.raises(ValueError):
        SeasonalNaiveForecaster().fit(inputs)


def test_unknown_variant_and_fresh_factory():
    with pytest.raises(ValueError, match="Unknown naive variant"):
        SeasonalNaiveForecaster("monthly")
    with pytest.raises(ValueError, match="Unknown naive variant"):
        make_seasonal_naive_factory("monthly")
    factory = make_seasonal_naive_factory("last_value")
    first, second = factory(), factory()
    assert first is not second
    first.fit(make_inputs({"A": [1]}))
    assert second._fitted_origin is None


@pytest.mark.parametrize("variant", VARIANTS)
def test_future_actuals_do_not_affect_fitted_origin_forecasts(tiny_panel, variant):
    panel, cfg = tiny_panel
    fold = development_splits(cfg)[0]
    first_inputs = construct_inputs(panel, fold)
    before = forecast(first_inputs, variant)
    changed = panel.copy(deep=True)
    changed.loc[changed.day_num.gt(fold.origin), "units_sold"] = 1_000_000
    after = forecast(construct_inputs(changed, fold), variant)
    pd.testing.assert_frame_equal(before, after)
    # Fitted state is a snapshot, independent even of later input-table mutations.
    model = SeasonalNaiveForecaster(variant).fit(first_inputs)
    first_inputs.history["units_sold"] = 99_999
    pd.testing.assert_frame_equal(model.predict(first_inputs), before)


@pytest.mark.parametrize("variant", VARIANTS)
def test_shared_backtest_saved_metrics_and_final_holdout_exclusion(tiny_panel, tmp_path, variant):
    panel, cfg = tiny_panel
    before = panel.copy(deep=True)
    directory = tmp_path / variant
    predictions, metrics, metadata = run_backtest(panel, cfg, make_seasonal_naive_factory(variant), directory)
    assert predictions.shape == (18, 10)
    assert set(predictions.origin) == {6, 9}
    assert predictions.groupby(["series_id", "origin"], observed=True).size().eq(3).all()
    assert metadata["max_target_day"] == 12
    assert metadata["final_holdout_evaluated"] is False
    assert validate_integrity(predictions, metadata, panel, cfg)["exact_grid"]
    scored = pd.read_parquet(directory / "scored_predictions.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(directory / "predictions.parquet"), predictions)
    assert "actual" in scored and scored.target_day_num.max() == 12
    expected = score_predictions(scored)
    pooled = metrics.loc[metrics.scope.eq("all_folds_pooled")].iloc[0]
    for name in ["mae", "rmse", "wape"]:
        assert pooled[name] == pytest.approx(expected[name])
    assert metrics[["pinball_q10", "pinball_q50", "pinball_q90"]].isna().all().all()
    altered = panel.copy(deep=True)
    altered.loc[altered.day_num.gt(12), "units_sold"] = 1_000_000
    other_predictions, other_metrics, _ = run_backtest(altered, cfg, make_seasonal_naive_factory(variant))
    pd.testing.assert_frame_equal(predictions, other_predictions)
    pd.testing.assert_frame_equal(metrics, other_metrics)
    pd.testing.assert_frame_equal(panel, before)
    for fold in development_splits(cfg):
        validate_predictions(predictions.loc[predictions.origin.eq(fold.origin)], construct_inputs(panel, fold).future_known)


def test_integrity_rejects_quantiles_missing_dates_and_holdout_metadata(tiny_panel):
    panel, cfg = tiny_panel
    predictions, _, metadata = run_backtest(panel, cfg, make_seasonal_naive_factory())
    for problem in ["quantiles", "dates", "metadata", "missing_row"]:
        changed, execution = predictions.copy(), dict(metadata)
        if problem == "quantiles":
            changed[["q10", "q50", "q90"]] = 0.0
        elif problem == "dates":
            changed.loc[0, "target_date"] += pd.Timedelta(days=1)
        elif problem == "metadata":
            execution["final_holdout_evaluated"] = True
        else:
            changed = changed.iloc[:-1]
        with pytest.raises(ValueError):
            validate_integrity(changed, execution, panel, cfg)


def test_reporting_preserves_ties_and_unavailable_wape():
    table = pd.DataFrame({"model": ["a", "b"], "scope": ["all_folds_pooled"] * 2,
                          "wape": [0.5, 0.5], "mae": [1, 2], "rmse": [2, 4]})
    assert "tie" in comparison_conclusion(table)
    table.loc[1, "wape"] = np.nan
    assert "unavailable" in comparison_conclusion(table)
    table.loc[1, "wape"] = 0.6
    assert comparison_conclusion(table).startswith("a has")


def test_experiment_orchestration_artifacts_figures_and_cutoff(tiny_project, monkeypatch):
    cfg, _ = tiny_project
    run_preprocessing(cfg)
    from scripts import run_seasonal_naive as runner
    original_loader = runner.load_panel
    cutoffs = []

    def guarded_loader(config, cutoff=None):
        cutoffs.append(cutoff)
        assert cutoff == 12
        return original_loader(config, cutoff=cutoff)

    monkeypatch.setattr(runner, "load_panel", guarded_loader)
    comparison = run_experiment(cfg, output_dir="m5_test")
    assert cutoffs == [12]
    assert len(comparison) == 6
    from pathlib import Path
    for variant in VARIANTS:
        destination = Path(cfg["paths"]["processed"]) / "m5_test" / variant
        for name in ["predictions.parquet", "scored_predictions.parquet", "metrics.csv",
                     "execution_metadata.json", "fit_summary.csv", "integrity.json"]:
            assert (destination / name).stat().st_size > 0
        assert len(pd.read_csv(destination / "fit_summary.csv")) == 6
    reports = Path(cfg["paths"]["reports"])
    assert len(list((reports / "figures").glob("m5_*.png"))) == 10
    assert (reports / "m5_baseline_comparison.csv").is_file()
    findings = (reports / "m5_seasonal_naive_findings.md").read_text()
    assert "were not loaded, evaluated, scored or plotted" in findings


def test_cli_models_config_and_output_dir(tiny_project, monkeypatch):
    cfg, _ = tiny_project
    import sys
    from scripts import run_seasonal_naive as runner
    seen = []
    monkeypatch.setattr(runner, "load_config", lambda path: seen.append(path) or cfg)
    monkeypatch.setattr(runner, "run_experiment", lambda config, models, directory: seen.append((config, models, directory)))
    monkeypatch.setattr(sys, "argv", ["run_seasonal_naive", "--config", "alternate.yaml",
                                      "--models", "last_value", "--output-dir", "custom_m5"])
    runner.main()
    assert seen == ["alternate.yaml", (cfg, ["last_value"], "custom_m5")]
