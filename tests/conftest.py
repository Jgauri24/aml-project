from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from src.config import load_config
from src.data.load_data import load_sales, load_calendar, load_prices
from src.data.preprocess import build_panel


@pytest.fixture
def tiny_project(tmp_path):
    cfg = deepcopy(load_config())
    for key in cfg["paths"]:
        directory = tmp_path / key
        directory.mkdir()
        cfg["paths"][key] = str(directory)
    cfg["selection"].update(cutoff_day=6, trailing_days=4, n_per_group=1)
    cfg["observation"].update(start_day=1, end_day=15)
    cfg["forecast"].update(horizon=3, development_origins=[6, 9], final_origin=12, final_end_day=15)
    cfg["eda"].update(cutoff_day=6, acf_lags=[1, 2], minimum_price_pairs=2)
    raw = tmp_path / "raw"
    dates = pd.date_range("2011-01-01", periods=18)
    calendar = pd.DataFrame({"date": dates, "wm_yr_wk": 11101 + np.arange(18) // 7,
                             "weekday": dates.day_name(), "wday": (np.arange(18) % 7) + 1,
                             "month": dates.month, "year": dates.year,
                             "d": [f"d_{i}" for i in range(1, 19)],
                             "event_name_1": [None] * 18, "event_type_1": [None] * 18,
                             "event_name_2": [None] * 18, "event_type_2": [None] * 18,
                             "snap_CA": np.arange(18) % 2, "snap_TX": np.arange(18) % 2,
                             "snap_WI": np.arange(18) % 2})
    calendar.to_csv(raw / "calendar.csv", index=False)
    sales = pd.DataFrame({"id": ["A_CA_1_evaluation", "B_TX_1_evaluation", "C_CA_1_evaluation"],
                          "item_id": ["A", "B", "C"], "store_id": ["CA_1", "TX_1", "CA_1"],
                          "dept_id": ["D1", "D2", "D1"], "cat_id": ["C1", "C2", "C1"],
                          "state_id": ["CA", "TX", "CA"]})
    for day in range(1, 16):
        sales[f"d_{day}"] = [0 if day < 8 else 1, day % 3, day * 2]
    sales.to_csv(raw / "sales_train_evaluation.csv", index=False)
    validation = sales.drop(columns=["d_13", "d_14", "d_15"]).copy()
    validation["id"] = validation.id.str.replace("_evaluation", "_validation")
    validation.to_csv(raw / "sales_train_validation.csv", index=False)
    prices = pd.DataFrame([{"item_id": item, "store_id": store, "wm_yr_wk": week, "sell_price": 1.0 + week % 3}
                           for item, store in zip(sales.item_id, sales.store_id)
                           for week in [11101, 11102, 11103] if not (item == "A" and week == 11101)])
    prices.to_csv(raw / "sell_prices.csv", index=False)
    submission = pd.DataFrame({"id": [f"{item}_{store}_{suffix}" for suffix in ["validation", "evaluation"]
                                      for item, store in zip(sales.item_id, sales.store_id)]})
    for step in range(1, 4):
        submission[f"F{step}"] = 0
    submission.to_csv(raw / "sample_submission.csv", index=False)
    return cfg, raw


@pytest.fixture
def tiny_panel(tiny_project):
    cfg, raw = tiny_project
    sales = load_sales(raw, 15)
    panel = build_panel(sales, load_calendar(raw), load_prices(raw), 1, 15)
    return panel, cfg
