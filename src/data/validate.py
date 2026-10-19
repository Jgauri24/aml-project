"""Fail-fast structural panel validation; report holdout structure, not target summaries."""
import numpy as np
import pandas as pd
from src.data.load_data import STATIC, assert_unique, require_columns


def validate_panel(panel, start_day, end_day, expected_series=None):
    require_columns(panel, ["series_id", *STATIC, "d", "day_num", "date", "units_sold",
                            "wm_yr_wk", "sell_price", "is_zero_sale", "price_missing",
                            "possible_prelaunch", "availability_unknown"], "panel")
    assert_unique(panel, ["series_id", "day_num"], "panel")
    assert_unique(panel, ["series_id", "date"], "panel dates")
    if panel[["series_id", *STATIC, "d", "day_num", "date", "wm_yr_wk"]].isna().any().any():
        raise ValueError("Missing indexed/static fields")
    n_series = panel["series_id"].nunique()
    if expected_series is not None and n_series != expected_series:
        raise ValueError("Unexpected series count")
    ordered = panel.sort_values(["series_id", "day_num"])
    counts = ordered.groupby("series_id", observed=True).size()
    expected_days = end_day - start_day + 1
    if not counts.eq(expected_days).all() or len(panel) != n_series * expected_days:
        raise ValueError("Incomplete daily panel: missing or excess rows")
    bounds = ordered.groupby("series_id", observed=True)["day_num"].agg(["min", "max"])
    if not bounds["min"].eq(start_day).all() or not bounds["max"].eq(end_day).all():
        raise ValueError("Incorrect observation bounds")
    diffs = ordered.groupby("series_id", observed=True)["day_num"].diff().dropna()
    date_diffs = ordered.groupby("series_id", observed=True)["date"].diff().dropna()
    if not diffs.eq(1).all() or not date_diffs.eq(pd.Timedelta(days=1)).all():
        raise ValueError("Dates/days are not continuous")
    day_dates = panel[["day_num", "date"]].drop_duplicates()
    if day_dates["day_num"].duplicated().any() or day_dates["date"].duplicated().any():
        raise ValueError("Day/date mapping differs across series")
    if not panel["d"].astype("string").eq("d_" + panel["day_num"].astype("string")).all():
        raise ValueError("Day label mismatch")
    if (panel.groupby("series_id", observed=True)[STATIC].nunique() != 1).any().any():
        raise ValueError("Static metadata changes within series")
    actual_ids = panel["item_id"].astype("string") + "__" + panel["store_id"].astype("string")
    if not actual_ids.eq(panel["series_id"].astype("string")).all():
        raise ValueError("series_id does not match item/store")
    y = panel["units_sold"].dropna().to_numpy()
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Negative or infinite targets")
    p = panel["sell_price"].dropna().to_numpy()
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Nonpositive or infinite observed prices")
    if not panel["price_missing"].eq(panel["sell_price"].isna()).all():
        raise ValueError("price_missing inconsistent with price")
    zeros = panel["units_sold"].eq(0).where(panel["units_sold"].notna(), pd.NA).astype("boolean")
    if not panel["is_zero_sale"].astype("boolean").equals(zeros):
        raise ValueError("Zero/missing target distinction lost")
    if not panel["availability_unknown"].all():
        raise ValueError("Availability is not observed in M5")
    return {
        "rows": int(len(panel)), "series": int(n_series), "days_per_series": expected_days,
        "day_start": start_day, "day_end": end_day,
        "date_start": str(panel["date"].min().date()), "date_end": str(panel["date"].max().date()),
        "duplicate_series_days": 0, "missing_indexed_rows": 0,
        "missing_targets": int(panel["units_sold"].isna().sum()),
        "checks": "unique keys, continuity, day/date mapping, static metadata, target/price domain, zero/missing flags",
        "holdout_policy": "Only mechanical integrity/missingness checks on holdout; no target distribution or performance analysis.",
    }
