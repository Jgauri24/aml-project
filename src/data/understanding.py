"""Structural source audit and dataset dictionary, with cutoff-safe target previews."""
import argparse
import gc
import json
import pandas as pd
from src.config import load_config, data_path
from src.data.load_data import (EXPECTED_FILES, SALES_META, STATIC, day_columns,
                                load_sales, load_calendar, load_prices, assert_unique)
from src.data.select_skus import run_selection


def markdown_table(frame):
    # Avoid an optional tabulate dependency.
    text = frame.astype("string").fillna("—")
    lines = ["| " + " | ".join(map(str, text.columns)) + " |", "| " + " | ".join(["---"] * len(text.columns)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in text.itertuples(index=False, name=None))
    return "\n".join(lines)


def compact_schema(columns, dtypes):
    days = day_columns(columns)
    forecasts = [c for c in columns if c.startswith("F") and c[1:].isdigit()]
    ordinary = [c for c in columns if c not in days + forecasts]
    rows = [{"column(s)": c, "dtype": str(dtypes[c])} for c in ordinary]
    for family in [days, forecasts]:
        if family:
            rows.append({"column(s)": f"{family[0]} … {family[-1]} (all consecutive)",
                         "dtype": "float32 (explicit target loader)" if days else str(dtypes[family[0]])})
    return rows


def audit_raw(cfg):
    raw, reports = data_path(cfg, "raw"), data_path(cfg, "reports")
    reports.mkdir(parents=True, exist_ok=True)
    cutoff = cfg["selection"]["cutoff_day"]
    report = {"target_inspection_cutoff": cutoff, "files": {}}
    calendar = load_calendar(raw)
    for name in EXPECTED_FILES:
        path = raw / name
        header = pd.read_csv(path, nrows=0)
        days = day_columns(header.columns)
        if days:
            safe = load_sales(raw, cutoff, filename=name)
            shape = [len(safe), len(header.columns)]
            unique = {c: int(safe[c].nunique()) for c in STATIC}
            unique["item_store_series"] = int(safe.series_id.nunique())
            missing = {"scope": f"metadata and target d_1–d_{cutoff} only",
                       "static_missing_cells": int(safe[SALES_META].isna().sum().sum()),
                       "target_missing_cells": int(safe[[f'd_{d}' for d in range(1, cutoff + 1)]].isna().sum().sum())}
            preview_columns = [*SALES_META, "d_1", "d_2", f"d_{cutoff}"]
            sample = safe[preview_columns].head(5)
            schema = compact_schema(header.columns, safe.dtypes)
            coverage = {"first_day": days[0], "last_day": days[-1],
                        "first_date": str(calendar.set_index("d").loc[days[0], "date"].date()),
                        "last_date": str(calendar.set_index("d").loc[days[-1], "date"].date())}
            del safe
            gc.collect()
        else:
            frame = load_prices(raw) if name == "sell_prices.csv" else pd.read_csv(path)
            if name == "sample_submission.csv":
                assert_unique(frame, ["id"], "submission")
                if set(frame.columns) != {"id", *[f"F{i}" for i in range(1, cfg["forecast"]["horizon"] + 1)]}:
                    raise ValueError("Unexpected submission schema")
            shape = list(frame.shape)
            unique = {c: int(frame[c].nunique()) for c in STATIC if c in frame}
            missing = {"scope": "all non-sales fields", "by_column": frame.isna().sum().astype(int).to_dict()}
            sample = frame.head(5)
            schema = compact_schema(frame.columns, frame.dtypes)
            coverage = {}
            if "wm_yr_wk" in frame:
                coverage = {"first_week_key": int(frame.wm_yr_wk.min()), "last_week_key": int(frame.wm_yr_wk.max()),
                            "distinct_week_keys": int(frame.wm_yr_wk.nunique())}
            if name == "calendar.csv":
                coverage.update({"first_date": str(frame.date.min()), "last_date": str(frame.date.max()),
                                 "first_day": "d_1", "last_day": f"d_{len(frame)}"})
            if name == "sample_submission.csv":
                unique["suffix_counts"] = frame.id.str.rsplit("_", n=1).str[-1].value_counts().to_dict()
            del frame
            gc.collect()
        report["files"][name] = {"shape": shape, "schema": schema, "unique": unique,
                                  "missing": missing, "coverage": coverage,
                                  "sample": json.loads(sample.to_json(orient="records"))}
    # This mandated integrity comparison uses only pre-holdout observed history.
    end = cfg["forecast"]["final_origin"]
    evaluation = load_sales(raw, end).sort_values("series_id").reset_index(drop=True)
    validation = load_sales(raw, end, filename="sales_train_validation.csv").sort_values("series_id").reset_index(drop=True)
    compare_columns = ["series_id", *STATIC, *[f"d_{d}" for d in range(1, end + 1)]]
    pd.testing.assert_frame_equal(evaluation[compare_columns], validation[compare_columns])
    report["overlap_check"] = {"matched": True, "series": len(evaluation), "days": end,
                               "cells_compared": len(evaluation) * end, "excluded": "id suffix differs; no holdout targets compared"}
    report["keys"] = {"sales": "item_id + store_id unique in both files",
                      "calendar": "d and date independently unique",
                      "prices": "item_id + store_id + wm_yr_wk unique"}
    (reports / "source_audit.json").write_text(json.dumps(report, indent=2))
    write_dictionary(report, cfg)
    return report


def write_dictionary(audit, cfg):
    descriptions = {
        "calendar.csv": "One calendar date/day. `wm_yr_wk` is an encoded Walmart week key shared with weekly prices. Events may be null (no listed event); this is distinct from a failed calendar join. SNAP columns indicate a state's calendar flag.",
        "sell_prices.csv": "One listed item/store/week price. Missing item/week records do not prove unavailability. A week key repeats for many item/store pairs.",
        "sales_train_validation.csv": "One item/store series; daily columns d_1–d_1913. The ID has a validation suffix. Used only for source-overlap verification, not appended to evaluation history.",
        "sales_train_evaluation.csv": "One item/store series; daily columns d_1–d_1941. The sole canonical observed-sales source. d_1914–d_1941 are mechanically stored/validated but never analyzed or development-scored.",
        "sample_submission.csv": "One series/suffix submission row with F1–F28 horizon placeholders. Validation and evaluation suffixes refer to different windows, not different item/store populations.",
    }
    lines = ["# M5 dataset dictionary", "", "Generated by `python -m src.data.understanding`. CSV has no intrinsic dtypes; schemas show the explicit loader types (sales/price) or pandas inference. Consecutive column ranges include all endpoints.", ""]
    for name, info in audit["files"].items():
        lines += [f"## {name}", "", descriptions[name], "", f"Shape: **{info['shape'][0]:,} × {info['shape'][1]:,}**.", "", markdown_table(pd.DataFrame(info["schema"])), "", f"Coverage: `{info['coverage']}`", "", f"Distinct identifiers: `{info['unique']}`", "", f"Missingness: `{info['missing']}`", ""]
    lines += ["## Keys and joins", "", f"All three candidate key constraints passed. Validation and evaluation match exactly in static metadata and {audit['overlap_check']['cells_compared']:,} overlapping target cells; literal IDs differ by suffix.", "", f"Overlap audit: `{audit['overlap_check']}`", "", "Sales day **column names** become `d` on reshape; join `d` to calendar (many sales rows to one calendar day). Then join `(item_id, store_id, wm_yr_wk)` to prices (many daily rows to one weekly price). Both joins enforce many-to-one cardinality. No unmatched calendar days are accepted.", "", "## Canonical panel", "", "`series_id = item_id + '__' + store_id` has no validation/evaluation suffix. `units_sold` is observed unit sales, not latent unconstrained demand. Each selected series has every indexed daily row. Observed zeros stay zero; missing targets stay missing. `date` is a naive daily timestamp; `day_num` is the integer in `d_N`. Static identifiers and event strings use categorical dtypes; day/year integers, float32 targets/prices and boolean flags keep memory bounded.", "", "`is_zero_sale` is nullable when the target is missing. `price_missing` means no known numeric listing for that row. `possible_prelaunch` is a past-only uncertain proxy: no positive sale and no price yet observed through that day. `availability_unknown` is always true: M5 has no stock inventory field. No confirmed stockout or launch label is inferred. Historical `snap` picks the appropriate state flag; original `snap_CA/TX/WI` are retained.", "", "## Selection", "", "Only d_1–d_1801 are inspected for selection and EDA. Rank by trailing 365-day mean, break ties by series_id, split into three rank terciles, then seeded round-robin sampling across store/category strata (80 per group, seed 42). Complete observed history is required by default; no positive-sales filter is applied, preserving sparse/all-zero candidates. The sample is deliberately stratified and is not population-representative. Exact eligibility, pool boundaries and compromises are in `data/interim/selection_metadata.json`."]
    (data_path(cfg, "reports") / "dataset_dictionary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    audit = audit_raw(cfg)
    selected, selection = run_selection(cfg)
    print({name: info["shape"] for name, info in audit["files"].items()})
    print(audit["overlap_check"])
    print(selected.groupby("volume_group").size().to_dict())
