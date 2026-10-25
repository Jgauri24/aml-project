"""Raw CSV readers. Filter series before reshaping and never concatenate sales files."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

STATIC = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]
SALES_META = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_COLUMNS = ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
                    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                    "snap_CA", "snap_TX", "snap_WI"]
PRICE_COLUMNS = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
EXPECTED_FILES = ["calendar.csv", "sell_prices.csv", "sales_train_validation.csv",
                  "sales_train_evaluation.csv"]


def require_columns(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def assert_unique(frame, keys, name):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{name}: null or duplicate key {keys}")


def day_columns(columns):
    return sorted([c for c in columns if re.fullmatch(r"d_\d+", c)], key=lambda c: int(c[2:]))


def add_series_id(frame):
    result = frame.copy()
    result["series_id"] = result["item_id"].astype("string") + "__" + result["store_id"].astype("string")
    return result


def load_sales(raw_dir, end_day, selected=None, filename="sales_train_evaluation.csv", chunksize=512):
    """Read only requested days; if selected, keep matching rows in each CSV chunk."""
    path = Path(raw_dir) / filename
    header = pd.read_csv(path, nrows=0)
    days = [f"d_{d}" for d in range(1, end_day + 1)]
    require_columns(header, SALES_META + days, filename)
    keep_ids = None
    if selected is not None:
        assert_unique(selected, ["item_id", "store_id"], "selected series")
        keep_ids = set(add_series_id(selected)["series_id"])
    pieces = []
    for chunk in pd.read_csv(path, usecols=SALES_META + days,
                             dtype={**{c: "string" for c in SALES_META}, **{c: "float32" for c in days}},
                             chunksize=chunksize):
        chunk = add_series_id(chunk)
        if keep_ids is not None:
            chunk = chunk.loc[chunk["series_id"].isin(keep_ids)]
        if len(chunk):
            pieces.append(chunk)
    if not pieces:
        raise ValueError("No sales series loaded")
    sales = pd.concat(pieces, ignore_index=True)
    assert_unique(sales, ["item_id", "store_id"], filename)
    if sales[STATIC + ["id"]].isna().any().any():
        raise ValueError("Missing static metadata")
    values = sales[days].to_numpy()
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("Targets must be nonnegative and finite or missing")
    if keep_ids is not None and set(sales["series_id"]) != keep_ids:
        raise ValueError("Some selected series are absent from raw sales")
    return sales


def load_calendar(raw_dir):
    frame = pd.read_csv(Path(raw_dir) / "calendar.csv", parse_dates=["date"])
    require_columns(frame, CALENDAR_COLUMNS, "calendar")
    assert_unique(frame, ["d"], "calendar")
    assert_unique(frame, ["date"], "calendar dates")
    if not frame["d"].str.fullmatch(r"d_\d+").all():
        raise ValueError("Invalid calendar day identifiers")
    frame["day_num"] = frame["d"].str[2:].astype("int16")
    frame = frame.sort_values("day_num").reset_index(drop=True)
    if not np.all(np.diff(frame["day_num"]) == 1):
        raise ValueError("Calendar day identifiers are not continuous")
    if not (frame["date"].diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Calendar dates are not continuous")
    required = [c for c in CALENDAR_COLUMNS if not c.startswith("event_")]
    if frame[required].isna().any().any():
        raise ValueError("Missing required calendar fields")
    if not frame[["snap_CA", "snap_TX", "snap_WI"]].isin([0, 1]).all().all():
        raise ValueError("SNAP indicators must be binary")
    return frame


def load_prices(raw_dir, selected=None, chunksize=500_000):
    path = Path(raw_dir) / "sell_prices.csv"
    require_columns(pd.read_csv(path, nrows=0), PRICE_COLUMNS, "prices")
    keep = None if selected is None else set(add_series_id(selected)["series_id"])
    pieces = []
    for chunk in pd.read_csv(path, dtype={"item_id": "string", "store_id": "string",
                                          "wm_yr_wk": "int32", "sell_price": "float32"}, chunksize=chunksize):
        if keep is not None:
            ids = chunk["item_id"] + "__" + chunk["store_id"]
            chunk = chunk.loc[ids.isin(keep)]
        pieces.append(chunk)
    prices = pd.concat(pieces, ignore_index=True)
    assert_unique(prices, ["item_id", "store_id", "wm_yr_wk"], "prices")
    observed = prices["sell_price"].dropna()
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("Observed selling prices must be positive and finite")
    return prices
