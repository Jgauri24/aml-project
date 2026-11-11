"""Deterministic, volume-stratified sampling using first-origin history only."""
import argparse
import json
import numpy as np
import pandas as pd
from src.config import load_config, data_path, config_hash
from src.data.load_data import load_sales, day_columns, STATIC


def selection_statistics(sales, cfg):
    settings = cfg["selection"]
    cutoff, trailing = settings["cutoff_day"], settings["trailing_days"]
    # Explicit projection makes future targets irrelevant even if passed by a caller.
    days = [f"d_{d}" for d in range(1, cutoff + 1)]
    recent = [f"d_{d}" for d in range(max(1, cutoff - trailing + 1), cutoff + 1)]
    values = sales[days]
    stats = sales[["series_id", *STATIC]].copy()
    stats["observed_fraction"] = values.notna().mean(axis=1)
    stats["mean_history"] = values.mean(axis=1)
    stats["mean_trailing_365"] = sales[recent].mean(axis=1)
    stats["zero_fraction"] = values.eq(0).sum(axis=1) / values.notna().sum(axis=1)
    stats["selection_cutoff"] = cutoff
    eligible = stats.loc[stats["observed_fraction"] >= settings["minimum_observed_fraction"]].copy()
    eligible = eligible.sort_values(["mean_trailing_365", "series_id"]).reset_index(drop=True)
    # Rank terciles give exact, disjoint pools even when many means tie.
    labels = settings["groups"]
    if labels != ["low", "medium", "high"]:
        raise ValueError("Expected ordered volume groups low, medium, high")
    eligible["volume_group"] = np.asarray(labels)[np.minimum(np.arange(len(eligible)) * 3 // len(eligible), 2)]
    return eligible


def select_series(sales, cfg):
    stats = selection_statistics(sales, cfg)
    rng = np.random.default_rng(cfg["seed"])
    chosen = []
    compromises = []
    for label in cfg["selection"]["groups"]:
        pool = stats.loc[stats["volume_group"].eq(label)].sort_values("series_id")
        n = cfg["selection"]["n_per_group"]
        if len(pool) < n:
            raise ValueError(f"{label}: only {len(pool)} eligible series for {n} slots")
        strata = []
        for _, stratum in pool.groupby(cfg["selection"]["balance_columns"], sort=True, observed=True):
            strata.append(list(rng.permutation(stratum.index)))
        # Round robin across shuffled store/category strata. Sparse strata can exhaust.
        picked = []
        while len(picked) < n:
            for i in rng.permutation(len(strata)):
                if strata[i] and len(picked) < n:
                    picked.append(strata[i].pop())
        part = stats.loc[picked].copy()
        chosen.append(part)
        if pool["store_id"].nunique() != part["store_id"].nunique() or pool["cat_id"].nunique() != part["cat_id"].nunique():
            compromises.append(f"{label}: not all eligible stores/categories represented")
    selected = pd.concat(chosen, ignore_index=True).sort_values(["volume_group", "series_id"]).reset_index(drop=True)
    report = {
        "seed": cfg["seed"], "cutoff_day": cfg["selection"]["cutoff_day"],
        "config_sha256": config_hash(cfg), "total_candidates": len(sales), "eligible": len(stats),
        "excluded": len(sales) - len(stats),
        "eligibility": "Configured minimum observed fraction through cutoff; no positive-volume filter.",
        "ranking": "Trailing 365-day mean ascending; series_id breaks ties; equal rank terciles.",
        "sampling": "Seeded random within store/category strata; round-robin strata per volume tercile.",
        "compromises": compromises or ["Equal group sizes and diversity deliberately change population proportions; no population representativeness claimed."],
        "groups": {label: {
            "candidates": int(stats["volume_group"].eq(label).sum()),
            "selected": int(selected["volume_group"].eq(label).sum()),
            "pool_min_mean": float(stats.loc[stats.volume_group.eq(label), "mean_trailing_365"].min()),
            "pool_max_mean": float(stats.loc[stats.volume_group.eq(label), "mean_trailing_365"].max()),
            "stores": int(selected.loc[selected.volume_group.eq(label), "store_id"].nunique()),
            "categories": int(selected.loc[selected.volume_group.eq(label), "cat_id"].nunique()),
        } for label in cfg["selection"]["groups"]},
    }
    return selected, report


def run_selection(cfg):
    sales = load_sales(data_path(cfg, "raw"), cfg["selection"]["cutoff_day"])
    selected, report = select_series(sales, cfg)
    destination = data_path(cfg, "interim")
    destination.mkdir(parents=True, exist_ok=True)
    selected.to_csv(destination / "selected_skus.csv", index=False)
    (destination / "selection_metadata.json").write_text(json.dumps(report, indent=2))
    return selected, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    selected, report = run_selection(load_config(args.config))
    print(json.dumps(report, indent=2))
