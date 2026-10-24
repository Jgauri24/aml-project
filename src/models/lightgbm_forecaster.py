"""LightGBM tabular forecaster for the M9 milestone.

Strategy
--------
Direct multi-horizon forecasting: one LightGBM model is trained for all 28
horizons simultaneously, with ``horizon`` supplied as an integer feature.
This is simpler than training 28 separate models while still being horizon-aware.

The model is trained on the full development history up to the fold origin.
Features come from ``FeatureEngineer`` (lag, rolling, static, calendar, price).

Hyperparameter tuning uses rolling-window validation: folds 1–3 are used to
select params; fold 4 predictions are the held-out evaluation.

Outputs (saved per run)
-----------------------
- ``predictions.parquet`` — PREDICTION_COLUMNS format, all 4 development folds
- ``scored_predictions.parquet`` — with actuals attached
- ``metrics.csv`` — fold-level and pooled MAE / RMSE / WAPE
- ``feature_importance.csv`` — gain-based importance averaged across folds
- ``execution_metadata.json`` — reproducibility metadata
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except OSError as exc:
    raise ImportError(
        "LightGBM failed to load its native library. "
        "On macOS install libomp first: `brew install libomp`."
    ) from exc

from src.evaluation.backtesting import (
    BaseForecaster, ForecastInputs, FUTURE_KEYS, PREDICTION_COLUMNS,
    attach_actuals, construct_inputs, run_backtest,
)
from src.evaluation.metrics import score_predictions
from src.evaluation.splits import development_splits
from src.features.feature_engineering import FeatureEngineer


# ── Default hyperparameters ─────────────────────────────────────────────────
DEFAULT_PARAMS: Dict = {
    "objective": "regression_l1",   # MAE objective (robust to sparse demand)
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

# Candidate grids for rolling-validation tuning (kept small — no over-engineering)
TUNING_GRID: List[Dict] = [
    {"num_leaves": 31,  "learning_rate": 0.05, "n_estimators": 500},
    {"num_leaves": 63,  "learning_rate": 0.05, "n_estimators": 500},
    {"num_leaves": 127, "learning_rate": 0.03, "n_estimators": 700},
]


# ── LightGBM forecaster ──────────────────────────────────────────────────────
class LightGBMForecaster(BaseForecaster):
    """Direct multi-horizon LightGBM forecaster.

    Uses ``FeatureEngineer`` to build leakage-safe features, then trains a
    single LGB model across all SKUs and horizons.  ``horizon`` is included as
    a numeric feature so the model learns horizon-specific patterns.

    Parameters
    ----------
    params:
        LightGBM parameters dict.  Merged with DEFAULT_PARAMS (explicit values
        win).
    """

    def __init__(self, params: Optional[Dict] = None) -> None:
        self._params = {**DEFAULT_PARAMS, **(params or {})}
        self._fe: Optional[FeatureEngineer] = None
        self._model: Optional[lgb.LGBMRegressor] = None
        self._fitted_origin: Optional[int] = None
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.fit_summary_: Optional[pd.DataFrame] = None

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, inputs: ForecastInputs) -> "LightGBMForecaster":
        """Train one LightGBM model on history up to ``inputs.origin``."""
        self._fe = FeatureEngineer()
        self._fe.fit(inputs)

        X_train, y_train = self._fe.build_train_features(inputs)

        # horizon=NaN during training; real horizon value supplied at prediction
        X_train = X_train.copy()
        X_train["horizon"] = np.nan

        params = {k: v for k, v in self._params.items()
                  if k not in ("early_stopping_rounds",)}
        early_stop = self._params.get("early_stopping_rounds", 50)
        n_est = params.pop("n_estimators", 500)

        self._model = lgb.LGBMRegressor(n_estimators=n_est, **params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(
                X_train, y_train,
                callbacks=[lgb.early_stopping(early_stop, verbose=False),
                           lgb.log_evaluation(period=-1)],
                eval_set=[(X_train, y_train)],
            )

        # Feature importance
        self.feature_importance_ = pd.DataFrame({
            "feature": X_train.columns.tolist(),
            "importance_gain": self._model.booster_.feature_importance(importance_type="gain"),
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

        self.fit_summary_ = pd.DataFrame([{
            "origin": inputs.origin,
            "n_series": inputs.history["series_id"].nunique(),
            "train_rows": len(X_train),
            "best_iteration": getattr(self._model, "best_iteration_", n_est),
            "params": json.dumps(params),
        }])

        self._fitted_origin = inputs.origin
        return self

    # ── predict ──────────────────────────────────────────────────────────────
    def predict(self, inputs: ForecastInputs) -> pd.DataFrame:
        """Predict all 28 horizons for all series using the fitted model."""
        if self._fitted_origin is None or self._fitted_origin != inputs.origin:
            raise RuntimeError("Call fit() for this origin before predict()")

        future = inputs.future_known[FUTURE_KEYS].copy()
        X_pred = self._fe.build_predict_features(inputs)

        # Supply real horizon for direct multi-step differentiation
        X_pred = X_pred.copy()
        X_pred["horizon"] = future["horizon"].values

        raw_preds = self._model.predict(X_pred)
        y_pred = np.maximum(raw_preds, 0.0)  # Clip negatives

        future = future.copy()
        future["model"] = "lightgbm"
        future["y_pred"] = y_pred
        future[["q10", "q50", "q90"]] = np.nan

        return future[PREDICTION_COLUMNS]


# ── Rolling-validation hyperparameter tuning ─────────────────────────────────
def tune_hyperparameters(
    panel: pd.DataFrame,
    cfg: dict,
    grid: List[Dict] = TUNING_GRID,
) -> Dict:
    """Select best params by evaluating on the first N-1 development folds.

    Uses WAPE as the selection criterion (consistent with M5 evaluation).
    The final fold is left untouched (serves as out-of-sample validation).
    """
    folds = development_splits(cfg)
    tuning_folds = folds[:-1]  # Leave last fold as holdout

    if not tuning_folds:
        print("Only one development fold — skipping tuning, using defaults.")
        return {}

    print(f"Tuning on {len(tuning_folds)} fold(s), evaluating {len(grid)} configs...")
    development = panel.loc[panel["day_num"].le(cfg["forecast"]["final_origin"])].copy()

    best_wape = float("inf")
    best_params: Dict = {}

    for idx, candidate in enumerate(grid):
        wapes = []
        for fold in tuning_folds:
            inputs = construct_inputs(development, fold)
            model = LightGBMForecaster(params=candidate)
            model.fit(inputs)
            pred = model.predict(inputs)
            scored = attach_actuals(pred, development)
            metrics = score_predictions(scored)
            wapes.append(metrics["wape"])

        mean_wape = float(np.nanmean(wapes))
        print(f"  Config {idx + 1}/{len(grid)}: {candidate}  ->  WAPE={mean_wape:.4f}")

        if mean_wape < best_wape:
            best_wape = mean_wape
            best_params = candidate

    print(f"Best config: {best_params}  (WAPE={best_wape:.4f})")
    return best_params


# ── Comparison helper ─────────────────────────────────────────────────────────
def compare_with_baselines(
    lgb_metrics: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Attempt to load seasonal-naive and Kalman metrics; return comparison table."""
    rows = [lgb_metrics]

    # 1. Try local data/processed metrics
    processed_dir = output_dir.parent if output_dir.name == "lightgbm" else output_dir
    found_subdirs = set()
    for subdir in [
        "seasonal_naive/weekly_seasonal",
        "seasonal_naive/last_value",
        "kalman_forecaster/local_level_seasonal",
        "kalman_forecaster/local_trend_seasonal",
    ]:
        candidate = processed_dir / subdir / "metrics.csv"
        if candidate.exists():
            df = pd.read_csv(candidate)
            rows.append(df)
            found_subdirs.add(subdir)

    # 2. Fall back to reports/ if baseline directories were not found
    reports_dir = output_dir.parent.parent.parent / "reports"
    if not reports_dir.exists():
        reports_dir = Path("reports")

    if not any("seasonal_naive" in s for s in found_subdirs) and reports_dir.exists():
        m5_file = reports_dir / "m5_baseline_comparison.csv"
        if m5_file.exists():
            m5_df = pd.read_csv(m5_file)
            rows.append(m5_df)

    if not any("kalman" in s for s in found_subdirs) and reports_dir.exists():
        kalman_file = reports_dir / "kalman_variant_comparison.csv"
        if kalman_file.exists():
            kalman_df = pd.read_csv(kalman_file)
            k_rows = []
            for _, r in kalman_df.iterrows():
                k_rows.append({
                    "model": r["model"],
                    "origin": np.nan,
                    "scope": "all_folds_pooled",
                    "n_rows": int(r.get("prediction_rows", 26880)),
                    "n_observed": int(r.get("observed_rows", 26880)),
                    "mae": float(r["MAE"]),
                    "rmse": float(r["RMSE"]),
                    "wape": float(r["WAPE"]),
                    "pinball_q10": float(r.get("pinball_q10", np.nan)),
                    "pinball_q50": float(r.get("pinball_q50", np.nan)),
                    "pinball_q90": float(r.get("pinball_q90", np.nan)),
                })
            rows.append(pd.DataFrame(k_rows))

    return pd.concat(rows, ignore_index=True) if len(rows) > 1 else lgb_metrics.copy()


# ── Main runner ───────────────────────────────────────────────────────────────
def run_lightgbm_forecaster(
    panel: pd.DataFrame,
    cfg: dict,
    output_dir: str = "lightgbm",
    tune: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run full rolling-backtest with optional tuning and save outputs."""
    from src.config import data_path

    destination = data_path(cfg, "processed") / output_dir
    destination.mkdir(parents=True, exist_ok=True)

    # Optional rolling-validation tuning
    best_params: Dict = {}
    if tune:
        best_params = tune_hyperparameters(panel, cfg)

    # Collect fitted instances to avoid redundant refitting
    fitted_models: List[LightGBMForecaster] = []

    def factory() -> LightGBMForecaster:
        m = LightGBMForecaster(params=best_params)
        fitted_models.append(m)
        return m

    started = perf_counter()
    predictions, metrics, metadata = run_backtest(
        panel, cfg, factory, output_dir=output_dir
    )
    elapsed = perf_counter() - started

    # Collect feature importance directly from the models fitted in backtest
    importance_records = []
    for model in fitted_models:
        if model.feature_importance_ is not None and model._fitted_origin is not None:
            df = model.feature_importance_.copy()
            df["origin"] = model._fitted_origin
            importance_records.append(df)

    if not importance_records:
        importance_records = _collect_importance(panel, cfg, best_params)

    if importance_records:
        importance_df = (
            pd.concat(importance_records, ignore_index=True)
            .groupby("feature", as_index=False)["importance_gain"]
            .mean()
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )
        importance_df.to_csv(destination / "feature_importance.csv", index=False)
        print("\nTop-15 features by gain:")
        print(importance_df.head(15).to_string(index=False))

    # Comparison table
    print("\n-- Metrics comparison --")
    comparison = compare_with_baselines(metrics, destination)
    comparison.to_csv(destination / "comparison_metrics.csv", index=False)
    # Also save to reports/ for repository-wide transparency
    repo_reports = destination.parent.parent.parent / "reports"
    if repo_reports.exists():
        comparison.to_csv(repo_reports / "m9_lightgbm_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    metadata["wall_seconds"] = elapsed
    metadata["best_params"] = best_params
    metadata["output_dir"] = str(destination)

    print(f"\nDone in {elapsed:.1f}s.  Outputs -> {destination}")
    return predictions, metrics, metadata


def _collect_importance(
    panel: pd.DataFrame, cfg: dict, params: Dict
) -> List[pd.DataFrame]:
    """Refit one model per fold solely to harvest feature importances."""
    development = panel.loc[panel["day_num"].le(cfg["forecast"]["final_origin"])].copy()
    records: List[pd.DataFrame] = []
    for fold in development_splits(cfg):
        inputs = construct_inputs(development, fold)
        model = LightGBMForecaster(params=params)
        model.fit(inputs)
        if model.feature_importance_ is not None:
            df = model.feature_importance_.copy()
            df["origin"] = fold.origin
            records.append(df)
    return records


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    from src.config import load_config
    from src.data.preprocess import load_panel

    parser = argparse.ArgumentParser(description="Train LightGBM forecaster (M9).")
    parser.add_argument("--config", default="experiment.yaml")
    parser.add_argument("--output-dir", default="lightgbm",
                        help="Sub-directory under data/processed/")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip rolling-validation hyperparameter tuning")
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel = load_panel(cfg, cutoff=cfg["forecast"]["final_origin"])
    print(f"Panel: {panel['series_id'].nunique()} series, "
          f"days 1-{panel['day_num'].max()}")

    predictions, metrics, metadata = run_lightgbm_forecaster(
        panel, cfg,
        output_dir=args.output_dir,
        tune=not args.no_tune,
    )
    print("\nFinal metrics:")
    print(metrics.to_string(index=False))
