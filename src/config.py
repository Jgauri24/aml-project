"""Load the one experiment configuration; paths never depend on notebook cwd."""
from pathlib import Path
import hashlib
import json
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "experiment.yaml"
    cfg = yaml.safe_load(path.read_text())
    selection, forecast, obs = cfg["selection"], cfg["forecast"], cfg["observation"]
    if forecast["horizon"] <= 0 or selection["n_per_group"] <= 0 or selection["trailing_days"] <= 0:
        raise ValueError("Horizon, group count and trailing window must be positive")
    if not 0 < selection["minimum_observed_fraction"] <= 1:
        raise ValueError("minimum_observed_fraction must be in (0, 1]")
    if cfg["metrics"]["zero_wape_denominator"] != "nan":
        raise ValueError("Only explicit NaN for zero WAPE denominator is supported")
    origins = forecast["development_origins"]
    if origins != sorted(set(origins)) or not origins:
        raise ValueError("Development origins must be distinct and increasing")
    if selection["cutoff_day"] > min(origins):
        raise ValueError("Selection cannot see beyond the first development origin")
    if cfg["eda"]["cutoff_day"] > selection["cutoff_day"]:
        raise ValueError("EDA cutoff exceeds the selection cutoff")
    if max(origins) + forecast["horizon"] > forecast["final_origin"]:
        raise ValueError("Development windows overlap the final holdout")
    if forecast["final_origin"] + forecast["horizon"] != forecast["final_end_day"]:
        raise ValueError("Final window must match the configured horizon")
    if obs["end_day"] != forecast["final_end_day"] or obs["start_day"] != 1:
        raise ValueError("Canonical panel must span day 1 through final_end_day")
    if cfg["availability"] != {"future_prices": "forbidden", "future_snap": "excluded"}:
        raise ValueError("Only the conservative implemented availability policy is supported")
    if cfg["metrics"]["aggregation"] != "pooled" or cfg["metrics"]["quantiles"] != [0.1, 0.5, 0.9]:
        raise ValueError("Implemented metrics are pooled with q=0.1, 0.5, 0.9")
    if not cfg["quality"]["possible_prelaunch_requires_both"]:
        raise ValueError("Only the documented past-only prelaunch rule is supported")
    return cfg


def data_path(cfg, key):
    return ROOT / cfg["paths"][key]


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
