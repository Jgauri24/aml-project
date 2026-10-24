"""Origin-safe inputs, a minimal forecaster contract, and development-only evaluation."""
from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import platform
from time import perf_counter
from pathlib import Path
from typing import Callable
from weakref import ref
import numpy as np
import pandas as pd
from src.config import load_config, data_path, config_hash
from src.data.load_data import STATIC, assert_unique, require_columns
from src.data.preprocess import add_quality_flags, load_panel
from src.evaluation.metrics import score_predictions
from src.evaluation.splits import Fold, development_splits

# Conservative allowlist. Future actual price, SNAP, quality flags and sales are absent.
KNOWN_CALENDAR = ["wm_yr_wk", "weekday", "wday", "month", "year",
                  "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
FUTURE_KEYS = ["series_id", "origin", "horizon", "target_day_num", "target_date"]
PREDICTION_COLUMNS = ["model", *FUTURE_KEYS, "y_pred", "q10", "q50", "q90"]


@dataclass
class ForecastInputs:
    history: pd.DataFrame
    static: pd.DataFrame
    future_known: pd.DataFrame
    origin: int
    horizon: int


class BaseForecaster(ABC):
    """Fit all learned transformations inside fit(), using inputs.history only."""
    @abstractmethod
    def fit(self, inputs: ForecastInputs):
        pass

    @abstractmethod
    def predict(self, inputs: ForecastInputs) -> pd.DataFrame:
        pass


class DummyForecaster(BaseForecaster):
    """Constant output only to test infrastructure. This is not a forecasting baseline."""
    def __init__(self, value=0.0):
        self.value = float(value)

    def fit(self, inputs):
        self.fitted = True
        return self

    def predict(self, inputs):
        if not getattr(self, "fitted", False):
            raise RuntimeError("Call fit before predict")
        prediction = inputs.future_known[FUTURE_KEYS].copy()
        prediction["model"] = "dummy_zero" if self.value == 0 else "dummy_constant"
        prediction["y_pred"] = self.value
        prediction[["q10", "q50", "q90"]] = np.nan
        return prediction[PREDICTION_COLUMNS]


def construct_inputs(panel, fold):
    """Project explicit allowlists; never give the forecaster the original panel."""
    historical_columns = ["series_id", *STATIC, "d", "day_num", "date", "units_sold", "sell_price",
                          *KNOWN_CALENDAR, "snap_CA", "snap_TX", "snap_WI", "snap"]
    require_columns(panel, historical_columns, "model input panel")
    history = panel.loc[panel["day_num"].le(fold.origin), historical_columns].copy()
    if history.empty or history["day_num"].max() != fold.origin:
        raise ValueError("Training history does not reach origin")
    # Recompute heuristics solely from this fold's history, never trust full-panel flags.
    history = add_quality_flags(history)
    assert_unique(history, ["series_id", "day_num"], "training history")
    counts = history.groupby("series_id", observed=True).size()
    if not counts.eq(fold.origin).all():
        raise ValueError("Training history must include every indexed day from 1 to origin")
    static = history[["series_id", *STATIC]].drop_duplicates().reset_index(drop=True)
    assert_unique(static, ["series_id"], "static metadata")
    # Derive future dates from origin, so future target rows are not needed as inputs.
    origin_date = history.loc[history.day_num.eq(fold.origin), "date"].unique()
    if len(origin_date) != 1:
        raise ValueError("Origin has inconsistent dates")
    calendar = panel.loc[panel.day_num.isin(fold.target_days), ["day_num", "date", *KNOWN_CALENDAR]].drop_duplicates()
    assert_unique(calendar, ["day_num"], "future known calendar")
    if set(calendar.day_num) != set(fold.target_days):
        raise ValueError("Known future calendar is incomplete")
    expected_dates = pd.Timestamp(origin_date[0]) + pd.to_timedelta(calendar.day_num - fold.origin, unit="D")
    if not calendar.date.eq(expected_dates).all():
        raise ValueError("Future dates do not match day identifiers")
    calendar = calendar.rename(columns={"day_num": "target_day_num", "date": "target_date"})
    future = static[["series_id"]].merge(calendar, how="cross")
    future["origin"] = fold.origin
    future["horizon"] = future["target_day_num"] - fold.origin
    future = future[[*FUTURE_KEYS, *KNOWN_CALENDAR]].sort_values(["series_id", "horizon"]).reset_index(drop=True)
    return ForecastInputs(history, static, future, fold.origin, fold.horizon)


def validate_predictions(prediction, expected, model_name=None):
    require_columns(prediction, PREDICTION_COLUMNS, "forecasts")
    if len(prediction) != len(expected):
        raise ValueError("Incorrect prediction row count")
    assert_unique(prediction, ["series_id", "origin", "horizon"], "forecasts")
    if prediction["model"].isna().any() or prediction["model"].nunique() != 1:
        raise ValueError("Predictions must identify one model")
    if model_name is not None and not prediction["model"].eq(model_name).all():
        raise ValueError("Model name changed across folds")
    matched = expected[FUTURE_KEYS].merge(prediction[FUTURE_KEYS], on=FUTURE_KEYS,
                                         how="outer", validate="one_to_one", indicator=True)
    if not matched["_merge"].eq("both").all():
        raise ValueError("Prediction keys, dates, or horizons differ from requested grid")
    if (prediction["y_pred"].isna().any() or not np.isfinite(prediction["y_pred"]).all()
            or prediction["y_pred"].lt(0).any()):
        raise ValueError("Point forecasts must be finite and nonnegative")
    quantiles = prediction[["q10", "q50", "q90"]]
    if not quantiles.isna().all().all():
        if not np.isfinite(quantiles.to_numpy(dtype=float)).all():
            raise ValueError("Supply all quantiles or leave all unavailable")
        if not (quantiles.q10.le(quantiles.q50) & quantiles.q50.le(quantiles.q90) & quantiles.q10.ge(0)).all():
            raise ValueError("Quantiles must be nonnegative and ordered q10 <= q50 <= q90")


def attach_actuals(prediction, panel):
    """Evaluator-only operation. Missing values remain NA; missing keys are an error."""
    actuals = panel[["series_id", "day_num", "date", "units_sold"]].rename(
        columns={"day_num": "target_day_num", "date": "target_date", "units_sold": "actual"})
    scored = prediction.merge(actuals, on=["series_id", "target_day_num", "target_date"],
                              how="left", validate="many_to_one", indicator=True)
    if not scored["_merge"].eq("both").all():
        raise ValueError("Unmatched evaluation target keys")
    return scored.drop(columns="_merge")


def run_backtest(panel, cfg, factory: Callable[[], BaseForecaster], output_dir=None):
    # Drop the final holdout before any construction, not just before scoring.
    development = panel.loc[panel.day_num.le(cfg["forecast"]["final_origin"])].copy()
    forecasts, scored_folds, metric_rows, execution = [], [], [], []
    models = []  # Weak references detect reuse without retaining fitted model weights.
    model_name = None
    for fold in development_splits(cfg):
        inputs = construct_inputs(development, fold)
        model = factory()
        if not isinstance(model, BaseForecaster) or any(model is old() for old in models):
            raise ValueError("Factory must return a fresh BaseForecaster per fold")
        models.append(ref(model))
        start = perf_counter()
        model.fit(inputs)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        prediction = model.predict(inputs)
        predict_seconds = perf_counter() - start
        # Use a fresh grid; model mutations of its own inputs cannot alter evaluation.
        expected = construct_inputs(development, fold).future_known
        validate_predictions(prediction, expected, model_name)
        model_name = prediction.model.iloc[0]
        prediction = prediction[PREDICTION_COLUMNS].copy()
        scored = attach_actuals(prediction, development)
        metric_rows.append({"model": model_name, "origin": fold.origin, "scope": "fold_pooled", **score_predictions(scored)})
        forecasts.append(prediction)
        scored_folds.append(scored)
        execution.append({"origin": fold.origin, "train_rows": len(inputs.history), "forecast_rows": len(prediction),
                          "fit_seconds": fit_seconds, "predict_seconds": predict_seconds})
        del model  # Release fold-local state before constructing the next model.
    predictions = pd.concat(forecasts, ignore_index=True)
    scored = pd.concat(scored_folds, ignore_index=True)
    metric_rows.append({"model": model_name, "origin": pd.NA, "scope": "all_folds_pooled", **score_predictions(scored)})
    metrics = pd.DataFrame(metric_rows)
    metadata = {"model": model_name, "purpose": "infrastructure test" if model_name.startswith("dummy") else "development evaluation",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "pandas": pd.__version__, "seed": cfg["seed"], "config_sha256": config_hash(cfg),
                "prediction_rows": len(predictions), "folds": execution,
                "max_target_day": int(predictions.target_day_num.max()), "final_holdout_evaluated": False,
                "aggregation": "pooled over nonmissing actuals; WAPE ratio (not percent); no quantiles -> NaN pinball",
                "availability": cfg["availability"]}
    if output_dir is not None:
        destination = Path(output_dir)
        if not destination.is_absolute():
            destination = data_path(cfg, "processed") / destination
        destination.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(destination / "predictions.parquet", index=False)
        scored.to_parquet(destination / "scored_predictions.parquet", index=False)
        metrics.to_csv(destination / "metrics.csv", index=False)
        (destination / "execution_metadata.json").write_text(json.dumps(metadata, indent=2))
    return predictions, metrics, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DummyForecaster on development folds only")
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    panel_path = data_path(cfg, "processed") / "demand_panel.parquet"
    panel = load_panel(cfg, cutoff=cfg["forecast"]["final_origin"])
    predictions, metrics, metadata = run_backtest(panel, cfg, DummyForecaster, output_dir="dummy_backtest")
    metadata["panel_sha256"] = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    (data_path(cfg, "processed") / "dummy_backtest/execution_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(metrics.to_string(index=False))
    print(f"Predictions: {predictions.shape}; final holdout evaluated: {metadata['final_holdout_evaluated']}")
