"""Small synthetic integration run for the four existing forecasting adapters."""
from copy import deepcopy

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.backtesting import attach_actuals
from src.evaluation.metrics import compute_metric_tables
from src.evaluation.model_suite import run_model_suite
from src.evaluation.model_comparison import compare_models, rank_models, save_comparison_results
from src.models.deepar_config import DeepARConfig
from src.models.deepar_forecaster import make_deepar_factory
from src.models.kalman_forecaster import make_kalman_factory
from src.models.lightgbm_forecaster import LightGBMForecaster
from src.models.seasonal_naive import make_seasonal_naive_factory


def make_panel() -> pd.DataFrame:
    """Create two SKU/store series with calendar dates and known covariates."""
    dates = pd.date_range("2024-01-01", periods=196, freq="D")
    rows = []
    for idx, (item, store, state, period, base) in enumerate([
        ("ITEM_001", "CA_1", "CA", 7, 2),
        ("ITEM_002", "TX_1", "TX", 5, 1),
    ]):
        for day, date in enumerate(dates, start=1):
            rows.append({
                "series_id": f"{item}__{store}", "item_id": item,
                "store_id": store, "dept_id": "DEPT_1", "cat_id": "FOODS",
                "state_id": state, "d": f"d_{day}", "day_num": day,
                "date": date,
                "units_sold": float(base + ((day + idx) % period == 0) + ((day % 7) < 2)),
                "sell_price": 1.0 + 0.05 * (day // 28), "wm_yr_wk": 12000 + day // 7,
                "weekday": date.day_name(),
                "wday": ((date.dayofweek + 2) % 7) + 1,
                "month": date.month, "year": date.year,
                "event_name_1": None, "event_type_1": None,
                "event_name_2": None, "event_type_2": None,
                "snap_CA": int(state == "CA" and day % 7 == 0),
                "snap_TX": int(state == "TX" and day % 7 == 0),
                "snap_WI": 0, "snap": int(day % 7 == 0),
            })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = deepcopy(load_config())
    cfg["forecast"].update(horizon=28, development_origins=[140],
                            final_origin=168, final_end_day=196)
    cfg["observation"]["end_day"] = 196
    cfg["selection"]["cutoff_day"] = 140
    cfg["eda"]["cutoff_day"] = 140
    cfg["backtest"] = {"window_mode": "expanding", "training_window_days": None,
                       "step_size": 28, "sku_ids": None, "store_ids": None}

    deepar_cfg = DeepARConfig(
        context_length=28, prediction_length=28, hidden_dim=8, num_layers=1,
        batch_size=2, epochs=1, early_stopping_patience=1, val_window_length=28,
        windows_per_series_train=2, num_samples=8, device="cpu", seed=42,
    )
    lightgbm_params = {
        "objective": "regression_l1", "metric": "mae", "n_estimators": 8,
        "num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 2,
        "feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 0,
        "n_jobs": 1, "random_state": 42, "verbose": -1,
    }
    factories = {
        "seasonal_naive": make_seasonal_naive_factory("weekly_seasonal"),
        "kalman": make_kalman_factory("local_level", fit_window=140, n_jobs=1),
        "lightgbm": lambda: LightGBMForecaster(lightgbm_params),
        "deepar": make_deepar_factory(deepar_cfg),
    }
    panel = make_panel()
    forecasts, metrics, metadata = run_model_suite(panel, cfg, factories)
    print(f"Models: {', '.join(forecasts)}")
    print(f"Shared rows per model: {next(iter(metadata.values()))['expected_forecast_rows']}")
    print(f"Shared origin/date: d_140 / {metadata[next(iter(metadata))]['origin_dates']['140']}")
    for name, frame in forecasts.items():
        has_quantiles = frame[["q10", "q50", "q90"]].notna().all().all()
        print(f"{name}: rows={len(frame)}, quantiles={bool(has_quantiles)}, dates={frame.target_date.nunique()}")
    print(metrics.loc[metrics.scope.eq("all_folds_pooled"),
                      ["model", "mae", "rmse", "wape"]].to_string(index=False))

    # Score evaluator-attached actuals through the reusable metric aggregation API.
    # The MASE scale is calculated from panel history through each forecast origin.
    scored = pd.concat([attach_actuals(frame, panel) for frame in forecasts.values()],
                       ignore_index=True)
    metric_tables = compute_metric_tables(
        scored,
        training_history=panel,
        series_metadata=panel[["series_id", "item_id", "store_id"]].drop_duplicates(),
    )
    print("\nReusable metric engine portfolio comparison:")
    columns = ["model", "mae", "rmse", "wape", "smape", "mase", "pinball_q10",
               "pinball_q50", "pinball_q90", "coverage_80", "mean_interval_width_80", "wis_80"]
    print(metric_tables["overall"][columns].to_string(index=False))

    requested_pairs = [("kalman_local_level", "lightgbm"),
                       ("kalman_local_level", "deepar"), ("lightgbm", "deepar")]
    requested_pairs.extend(("naive_weekly_seasonal", advanced)
                           for advanced in ("kalman_local_level", "lightgbm", "deepar"))
    comparisons = compare_models(scored, requested_pairs, horizon=28, origin_step=28,
                                 n_bootstrap=500, seed=42)
    demand = panel.groupby("series_id", as_index=False).units_sold.mean()
    cutoff = demand.units_sold.median()
    segments = dict(zip(demand.series_id,
                        np.where(demand.units_sold.le(cutoff), "low", "high")))
    execution_rows = []
    for name, info in metadata.items():
        for fold in info["folds"]:
            execution_rows.append({"model": name, **fold,
                                   "total_seconds": fold["fit_seconds"] + fold["predict_seconds"]})
    rankings = rank_models(scored, series_metadata=panel[["series_id", "item_id", "store_id"]].drop_duplicates(),
                           demand_segments=segments, execution=pd.DataFrame(execution_rows))
    output_paths = save_comparison_results({**comparisons, **rankings},
                                           "reports/model_comparison_demo", file_format="csv")
    print("\nDM test status (one origin is insufficient for significance testing):")
    print(comparisons["dm_tests"][["model_a", "model_b", "loss", "mean_loss_difference_a_minus_b",
                                   "p_value", "status"]].to_string(index=False))
    print("\nMAE cluster-bootstrap differences (model A minus model B):")
    print(comparisons["bootstrap"].loc[comparisons["bootstrap"].metric.eq("mae"),
                                       ["model_a", "model_b", "difference_a_minus_b", "ci_lower",
                                        "ci_upper", "zero_in_ci", "n_bootstrap"]].to_string(index=False))
    print(f"Saved {len(output_paths)} comparison/ranking tables to reports/model_comparison_demo/")


if __name__ == "__main__":
    main()
