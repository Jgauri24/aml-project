"""Global direct-horizon LightGBM with origin-safe training and reloadable folds.

Historical training blocks follow the configured forecast cadence. All horizons
are observed in training; lag availability and frozen origin features match
inference. Tuning uses pooled WAPE on all but the last development fold. The
reserved final holdout is never evaluated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except OSError as exc:
    raise ImportError("LightGBM native library could not load; on macOS provide libomp.") from exc

from src.config import ROOT, config_hash, data_path, load_config
from src.evaluation.backtesting import (
    BaseForecaster, ForecastInputs, FUTURE_KEYS, PREDICTION_COLUMNS,
    attach_actuals, construct_inputs, run_backtest,
)
from src.evaluation.metrics import score_predictions
from src.evaluation.splits import development_splits
from src.features.feature_engineering import FEATURE_VERSION, TRAINING_STRATEGY, FeatureEngineer

DEFAULT_PARAMS = {
    "objective": "regression_l1", "metric": "mae", "num_leaves": 63,
    "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "min_child_samples": 20, "n_estimators": 500,
    "verbose": -1, "n_jobs": -1, "random_state": 42,
}
TUNING_GRID = [
    {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 500},
    {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 500},
    {"num_leaves": 127, "learning_rate": 0.03, "n_estimators": 700},
]


def _json_safe(value):
    """Unavailable metrics become JSON null; retain numeric values elsewhere."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


class LightGBMForecaster(BaseForecaster):
    """One pooled model per fold, trained on real historical origin/horizon rows."""
    def __init__(self, params: dict | None = None) -> None:
        self._params = {**DEFAULT_PARAMS, **(params or {})}
        if self._params.get("early_stopping_rounds"):
            raise ValueError("Use fixed n_estimators selected by rolling validation; no training-set early stopping")
        self._params.pop("early_stopping_rounds", None)
        self._fe = None
        self._model = None
        self._fitted_origin = None
        self._training_horizon = None
        self._training_origins = []
        self.feature_importance_ = None
        self.fit_summary_ = None

    def fit(self, inputs: ForecastInputs) -> LightGBMForecaster:
        self._fitted_origin = None
        self._fe = FeatureEngineer().fit(inputs)
        X, y, keys = self._fe.build_direct_train_features(inputs)
        if set(keys.horizon) != set(range(1, inputs.horizon + 1)):
            raise ValueError("Direct model needs observed training labels for every configured horizon")
        self._model = lgb.LGBMRegressor(**self._params)
        # n_estimators is chosen by chronological validation. Training data is
        # never presented as a pretend early-stopping validation set.
        self._model.fit(X, y, callbacks=[lgb.log_evaluation(period=0)])
        self._training_horizon = inputs.horizon
        self._training_origins = sorted(map(int, keys.origin.unique()))
        self._fitted_origin = inputs.origin
        self._update_importance()
        self.fit_summary_ = pd.DataFrame([{
            "origin": inputs.origin, "n_series": inputs.history.series_id.nunique(),
            "train_rows": len(X), "training_origin_count": len(self._training_origins),
            "min_training_origin": min(self._training_origins),
            "max_training_target": int(keys.target_day_num.max()),
            "training_horizon": inputs.horizon,
            "observed_horizons": json.dumps(sorted(map(int, keys.horizon.unique()))),
            "best_iteration": self._booster().current_iteration(),
            "params": json.dumps(self._params),
        }])
        return self

    def _booster(self) -> lgb.Booster:
        if self._model is None:
            raise RuntimeError("Call fit() before accessing the booster")
        return self._model if isinstance(self._model, lgb.Booster) else self._model.booster_

    def _update_importance(self) -> None:
        booster = self._booster()
        self.feature_importance_ = pd.DataFrame({
            "feature": booster.feature_name(),
            "importance_gain": booster.feature_importance(importance_type="gain"),
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    def predict(self, inputs: ForecastInputs) -> pd.DataFrame:
        if self._fitted_origin is None or self._fitted_origin != inputs.origin:
            raise RuntimeError("Call fit() for this origin before predict()")
        if inputs.horizon > self._training_horizon:
            raise ValueError("Requested horizon exceeds the trained horizon")
        future = inputs.future_known[FUTURE_KEYS].copy()
        X = self._fe.build_predict_features(inputs)
        X["horizon"] = future.horizon.to_numpy(dtype="int16")
        if list(X) != self._booster().feature_name():
            raise ValueError("Prediction feature schema differs from the fitted booster")
        future["model"] = "lightgbm"
        future["y_pred"] = np.maximum(self._model.predict(X), 0.0)
        future[["q10", "q50", "q90"]] = np.nan
        return future[PREDICTION_COLUMNS]

    def save(self, directory: str | Path) -> None:
        """Native booster plus JSON feature state, sufficient for same-origin inference."""
        if self._fitted_origin is None:
            raise RuntimeError("Call fit() before saving a model")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_file = directory / "model.txt"
        self._booster().save_model(str(model_file))
        state = {"version": 1, "model": "lightgbm", "origin": int(self._fitted_origin),
                 "training_horizon": self._training_horizon, "training_origins": self._training_origins,
                 "strategy": TRAINING_STRATEGY, "params": self._params, "lightgbm": lgb.__version__,
                 "feature_state": self._fe.to_state(),
                 "fit_summary": self.fit_summary_.to_dict("records"),
                 "model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest()}
        (directory / "state.json").write_text(json.dumps(_json_safe(state), indent=2, allow_nan=False))

    @classmethod
    def load(cls, directory: str | Path) -> LightGBMForecaster:
        """Restore a fold without fitting or reading raw/processed actual sales."""
        directory = Path(directory)
        state = json.loads((directory / "state.json").read_text())
        if state.get("version") != 1 or state.get("strategy") != TRAINING_STRATEGY:
            raise ValueError("Unsupported saved model format or strategy")
        model_file = directory / "model.txt"
        if hashlib.sha256(model_file.read_bytes()).hexdigest() != state["model_sha256"]:
            raise ValueError("Saved booster does not match its feature state")
        model = cls(state["params"])
        model._model = lgb.Booster(model_file=str(model_file))
        model._fe = FeatureEngineer.from_state(state["feature_state"])
        model._fitted_origin = int(state["origin"])
        model._training_horizon = int(state["training_horizon"])
        model._training_origins = state["training_origins"]
        if model._fe._fitted_origin != model._fitted_origin:
            raise ValueError("Saved booster and feature origins differ")
        if model._booster().feature_name() != [*model._fe.feature_columns, "horizon"]:
            raise ValueError("Saved booster and feature schema differ")
        model.fit_summary_ = pd.DataFrame(state["fit_summary"])
        model._update_importance()
        return model


def tune_hyperparameters(panel: pd.DataFrame, cfg: dict, grid=None,
                         records: list | None = None, base_params: dict | None = None) -> dict:
    """Select using shared pooled WAPE on the first N-1 development folds only."""
    folds = development_splits(cfg)[:-1]
    if not folds:
        return {}
    grid = TUNING_GRID if grid is None else grid
    if not grid:
        raise ValueError("Tuning grid must contain a candidate")
    development = panel.loc[panel.day_num.le(cfg["forecast"]["final_origin"])].copy()
    best_wape, best_params = float("inf"), None
    print(f"Tuning {len(grid)} candidates on origins {[fold.origin for fold in folds]}", flush=True)
    for index, candidate in enumerate(grid):
        started = perf_counter()
        scored_folds, fold_metrics = [], []
        for fold in folds:
            inputs = construct_inputs(development, fold)
            model = LightGBMForecaster({**(base_params or {}), **candidate, "random_state": cfg["seed"]})
            model.fit(inputs)
            scored = attach_actuals(model.predict(inputs), development)
            scored_folds.append(scored)
            fold_metrics.append({"origin": fold.origin, **score_predictions(scored)})
        pooled = score_predictions(pd.concat(scored_folds, ignore_index=True))
        record = {"candidate": index + 1, "params": dict(candidate), "fold_metrics": fold_metrics,
                  "pooled_metrics": pooled, "seconds": perf_counter() - started}
        if records is not None:
            records.append(record)
        print(f"Candidate {index + 1}: {candidate}; pooled tuning WAPE={pooled['wape']:.6f}", flush=True)
        if np.isfinite(pooled["wape"]) and pooled["wape"] < best_wape:
            best_wape, best_params = pooled["wape"], dict(candidate)
        del model
    if best_params is None:
        raise ValueError("No candidate has a finite tuning WAPE; no winner selected")
    return best_params


def compare_with_baselines(lgb_metrics: pd.DataFrame, output_dir: Path, cfg: dict | None = None) -> pd.DataFrame:
    """Read all available comparable baseline metrics; never refit baselines."""
    cfg = load_config() if cfg is None else cfg
    processed, reports = data_path(cfg, "processed"), data_path(cfg, "reports")
    rows = [lgb_metrics]
    runs = ["seasonal_naive/last_value", "seasonal_naive/weekly_seasonal",
            "kalman_single_sku/run_001", "verification_pre_push/kalman_single_sku",
            *[f"kalman_forecaster/{variant}" for variant in
              ["local_level", "local_level_seasonal", "local_trend", "local_trend_seasonal"]]]
    for run in runs:
        metrics_file = processed / run / "metrics.csv"
        meta_file = processed / run / "execution_metadata.json"
        if metrics_file.is_file() and meta_file.is_file():
            meta = json.loads(meta_file.read_text())
            if meta.get("config_sha256") == config_hash(cfg) and meta.get("final_holdout_evaluated") is False:
                rows.append(pd.read_csv(metrics_file))
    found = set(pd.concat(rows, ignore_index=True).model)
    # Report-only fallbacks are limited to the unchanged standard experiment.
    if config_hash(cfg) == config_hash(load_config()):
        path = reports / "m5_baseline_comparison.csv"
        if path.is_file():
            table = pd.read_csv(path)
            rows.append(table.loc[~table.model.isin(found)])
        path = reports / "kalman_variant_comparison.csv"
        if path.is_file():
            for row in pd.read_csv(path).to_dict("records"):
                if row["model"] not in found:
                    rows.append(pd.DataFrame([{"model": row["model"], "origin": np.nan,
                        "scope": "all_folds_pooled", "n_rows": row["prediction_rows"],
                        "n_observed": row["observed_rows"],
                        **{key: row[key.upper()] for key in ["mae", "rmse", "wape"]},
                        **{key: row[key] for key in ["pinball_q10", "pinball_q50", "pinball_q90"]}}]))
    return pd.concat(rows, ignore_index=True).drop_duplicates(["model", "origin", "scope"]).reset_index(drop=True)


def run_lightgbm_forecaster(panel: pd.DataFrame, cfg: dict, output_dir: str = "lightgbm",
                            tune: bool = True, params: dict | None = None,
                            tuning_grid: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Shared rolling backtest with persisted tuning, timers and native model bundles."""
    started = perf_counter()
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = data_path(cfg, "processed") / destination
    destination.mkdir(parents=True, exist_ok=True)
    tuning_records = []
    tuning_started = perf_counter()
    selected = tune_hyperparameters(panel, cfg, tuning_grid, tuning_records, params) if tune else {}
    tuning_seconds = perf_counter() - tuning_started
    resolved = {**DEFAULT_PARAMS, **(params or {}), **selected, "random_state": cfg["seed"]}
    importance_records, summaries, artifacts = [], [], []

    class RecordingForecaster(LightGBMForecaster):
        def fit(self, inputs: ForecastInputs):
            super().fit(inputs)
            relative = Path("models") / f"origin_{inputs.origin}"
            self.save(destination / relative)
            artifacts.append({"origin": inputs.origin, "directory": str(relative),
                              "model": str(relative / "model.txt"), "state": str(relative / "state.json")})
            importance_records.append(self.feature_importance_.assign(origin=inputs.origin))
            summaries.append(self.fit_summary_.copy())
            return self

    backtest_started = perf_counter()
    predictions, metrics, metadata = run_backtest(panel, cfg, lambda: RecordingForecaster(resolved),
                                                 output_dir=destination)
    backtest_seconds = perf_counter() - backtest_started
    importance = pd.concat(importance_records, ignore_index=True).groupby("feature", as_index=False).importance_gain.mean()
    importance = importance.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    importance.to_csv(destination / "feature_importance.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(destination / "fit_summary.csv", index=False)
    comparison = compare_with_baselines(metrics, destination, cfg)
    comparison.to_csv(destination / "comparison_metrics.csv", index=False)
    reports = data_path(cfg, "reports")
    reports.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports / "m9_lightgbm_comparison.csv", index=False)
    (destination / "tuning_results.json").write_text(json.dumps(_json_safe(tuning_records), indent=2, allow_nan=False))
    metadata.update(best_params=resolved, selected_candidate=selected, tuning_enabled=tune,
                    tuning_origins=[fold.origin for fold in development_splits(cfg)[:-1]] if tune else [],
                    selection_metric="pooled WAPE over tuning folds; exact ties keep first candidate",
                    tuning_seconds=tuning_seconds, backtest_seconds=backtest_seconds,
                    wall_seconds=perf_counter() - started, output_dir=str(destination),
                    strategy=TRAINING_STRATEGY, feature_version=FEATURE_VERSION,
                    lightgbm=lgb.__version__, numpy=np.__version__,
                    feature_columns=[*FeatureEngineer().feature_columns, "horizon"],
                    training_horizon=cfg["forecast"]["horizon"], model_artifacts=artifacts,
                    tuning_results="tuning_results.json")
    # Persist the enriched object after tuning and backtesting have both finished.
    (destination / "execution_metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2, allow_nan=False))
    print(metrics[["model", "origin", "scope", "mae", "rmse", "wape"]].to_string(index=False), flush=True)
    print(f"Done in {metadata['wall_seconds']:.1f}s including tuning; outputs: {destination}", flush=True)
    return predictions, metrics, metadata


if __name__ == "__main__":
    from scripts.run_lightgbm_forecaster import main
    main()
