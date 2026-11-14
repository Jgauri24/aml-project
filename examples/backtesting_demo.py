"""Run the shared rolling backtester on two tiny synthetic SKU/store series."""
from copy import deepcopy

import pandas as pd

from src.config import load_config
from src.evaluation.backtesting import run_backtest
from src.models.seasonal_naive import make_seasonal_naive_factory


def synthetic_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=28, freq="D")
    records = []
    for item_id, store_id, period in [("SKU_A", "CA_1", 5), ("SKU_B", "TX_1", 3)]:
        for day, date in enumerate(dates, start=1):
            records.append({
                "series_id": f"{item_id}__{store_id}", "item_id": item_id,
                "store_id": store_id, "dept_id": "DEPT_1", "cat_id": "FOODS",
                "state_id": store_id.split("_")[0], "d": f"d_{day}", "day_num": day,
                "date": date, "units_sold": float((day * (1 if period == 5 else 2)) % period),
                "sell_price": 1.0, "wm_yr_wk": 12000 + day // 7,
                "weekday": date.day_name(), "wday": (day - 1) % 7 + 1,
                "month": date.month, "year": date.year,
                "event_name_1": None, "event_type_1": None,
                "event_name_2": None, "event_type_2": None,
                "snap_CA": 0, "snap_TX": 0, "snap_WI": 0, "snap": 0,
            })
    return pd.DataFrame(records)


def main() -> None:
    cfg = deepcopy(load_config())
    cfg["selection"]["cutoff_day"] = 7
    cfg["eda"]["cutoff_day"] = 7
    cfg["forecast"].update(horizon=7, development_origins=[7, 14],
                            final_origin=21, final_end_day=28)
    cfg["observation"]["end_day"] = 28
    cfg["backtest"] = {"window_mode": "expanding", "training_window_days": None,
                       "step_size": 7, "sku_ids": None, "store_ids": None}
    predictions, metrics, metadata = run_backtest(
        synthetic_panel(), cfg, make_seasonal_naive_factory("weekly_seasonal"))
    print(f"Synthetic series: {metadata['series_count']}; origins: {list(metadata['origin_dates'].values())}")
    print(f"Predictions: {len(predictions)}; development maximum day: {metadata['max_target_day']}")
    print(metrics[["model", "origin", "scope", "mae", "rmse", "wape"]].to_string(index=False))


if __name__ == "__main__":
    main()
