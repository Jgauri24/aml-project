"""Build the canonical panel. CLI: python -m src.data.preprocess [--config ...]."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import numpy as np
import pandas as pd
from src.config import load_config, data_path, config_hash
from src.data.load_data import load_sales, load_calendar, load_prices, day_columns, STATIC
from src.data.select_skus import run_selection
from src.data.validate import validate_panel


def add_quality_flags(panel):
    """Past-only flags at each timestamp, never the first sale observed in all history.

    No inventory observations exist: availability_unknown is always true. No stockout
    label is manufactured. Possible prelaunch requires no positive sale and no price
    observed up to and including the timestamp; missing targets remain unknown.
    """
    result = panel.sort_values(["series_id", "day_num"]).reset_index(drop=True).copy()
    result["is_zero_sale"] = result["units_sold"].eq(0).where(result["units_sold"].notna(), pd.NA).astype("boolean")
    result["price_missing"] = result["sell_price"].isna()
    sale_seen = result["units_sold"].gt(0).fillna(False).groupby(result["series_id"], observed=True).cummax()
    price_seen = result["sell_price"].notna().groupby(result["series_id"], observed=True).cummax()
    result["possible_prelaunch"] = ~sale_seen & ~price_seen
    result["availability_unknown"] = True
    return result


def build_panel(sales, calendar, prices, start_day=1, end_day=1941):
    days = [f"d_{d}" for d in range(start_day, end_day + 1)]
    panel = sales[["series_id", *STATIC, *days]].melt(
        id_vars=["series_id", *STATIC], value_vars=days, var_name="d", value_name="units_sold")
    panel["day_num"] = panel["d"].str[2:].astype("int16")
    panel = panel.merge(calendar.drop(columns="day_num"), on="d", how="left",
                        validate="many_to_one", indicator="calendar_match")
    if not panel["calendar_match"].eq("both").all():
        raise ValueError("Unmatched calendar days")
    panel = panel.drop(columns="calendar_match")
    panel = panel.merge(prices, on=["item_id", "store_id", "wm_yr_wk"], how="left", validate="many_to_one")
    # State-specific SNAP for each series, alongside the original three flags.
    snap = np.zeros(len(panel), dtype="uint8")
    if not panel["state_id"].isin(["CA", "TX", "WI"]).all():
        raise ValueError("Unsupported state for SNAP mapping")
    for state in ["CA", "TX", "WI"]:
        mask = panel["state_id"].eq(state)
        snap[mask.to_numpy()] = panel.loc[mask, f"snap_{state}"].to_numpy(dtype="uint8")
    panel["snap"] = snap
    panel = add_quality_flags(panel)
    for column in ["series_id", *STATIC, "d", "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        # String categories keep all-null event columns categorical through Arrow.
        panel[column] = panel[column].astype("str").astype("category")
    for column in ["wday", "month", "snap_CA", "snap_TX", "snap_WI", "snap"]:
        panel[column] = panel[column].astype("uint8")
    panel["year"] = panel["year"].astype("int16")
    panel["wm_yr_wk"] = panel["wm_yr_wk"].astype("int32")
    panel["units_sold"] = panel["units_sold"].astype("float32")
    panel["sell_price"] = panel["sell_price"].astype("float32")
    return panel


def run_preprocessing(cfg):
    interim, processed, raw = (data_path(cfg, key) for key in ["interim", "processed", "raw"])
    processed.mkdir(parents=True, exist_ok=True)
    # Recompute selection from safe history; never trust an unrelated/stale selected CSV.
    selected, selection_report = run_selection(cfg)
    sales = load_sales(raw, cfg["observation"]["end_day"], selected=selected)
    panel = build_panel(sales, load_calendar(raw), load_prices(raw, selected), **{
        "start_day": cfg["observation"]["start_day"], "end_day": cfg["observation"]["end_day"]})
    validation = validate_panel(panel, **{
        "start_day": cfg["observation"]["start_day"], "end_day": cfg["observation"]["end_day"],
        "expected_series": len(selected)})
    panel.to_parquet(processed / "demand_panel.parquet", index=False, engine="pyarrow", compression="zstd")
    raw_hashes = {}
    for path in sorted(raw.glob("*.csv")):
        with path.open("rb") as stream:
            raw_hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "pandas": pd.__version__,
        "config_sha256": config_hash(cfg), "source_sha256": raw_hashes,
        "selection_cutoff": cfg["selection"]["cutoff_day"],
        "selection_csv_sha256": hashlib.sha256((interim / "selected_skus.csv").read_bytes()).hexdigest(),
        "source_sales": "sales_train_evaluation.csv only; no concatenation with validation",
        "target": "Observed unit sales, not latent demand; zeros preserved; missing targets never filled",
        "quality_flags": "Past-only at each row timestamp; availability always unknown; no stockout labels",
        "validation": validation, "dtypes": panel.dtypes.astype(str).to_dict(),
        "memory_bytes": int(panel.memory_usage(deep=True).sum()),
        "selection": selection_report,
    }
    (processed / "panel_metadata.json").write_text(json.dumps(metadata, indent=2))
    return panel, metadata


def load_panel(cfg, cutoff=None):
    """Reject stale artifacts; optionally load only the cutoff-safe analytical rows."""
    processed, interim = data_path(cfg, "processed"), data_path(cfg, "interim")
    metadata = json.loads((processed / "panel_metadata.json").read_text())
    if metadata["config_sha256"] != config_hash(cfg):
        raise ValueError("Processed panel configuration differs; rerun preprocessing")
    selected_hash = hashlib.sha256((interim / "selected_skus.csv").read_bytes()).hexdigest()
    if metadata["selection_csv_sha256"] != selected_hash:
        raise ValueError("Selected series file changed; rerun preprocessing")
    filters = None if cutoff is None else [("day_num", "<=", cutoff)]
    return pd.read_parquet(processed / "demand_panel.parquet", filters=filters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    _, metadata = run_preprocessing(load_config(args.config))
    print(json.dumps(metadata["validation"], indent=2))
