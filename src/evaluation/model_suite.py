"""Run existing model adapters on one shared rolling-origin forecast grid."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from src.evaluation.backtesting import (
    BaseForecaster,
    FUTURE_KEYS,
    PREDICTION_COLUMNS,
    construct_inputs,
    run_backtest,
)
from src.evaluation.splits import development_splits


def run_model_suite(
    panel: pd.DataFrame,
    cfg: dict,
    factories: Mapping[str, Callable[[], BaseForecaster]],
    output_dir: str | Path | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Backtest each supplied factory and verify exact fold/grid comparability.

    Factories must construct fresh adapters. Each model continues to own its
    training strategy; this runner only aligns protocol, execution, and output.
    """
    if not factories:
        raise ValueError("Provide at least one model factory")

    expected_parts = []
    development = panel.loc[panel.day_num.le(cfg["forecast"]["final_origin"])].copy()
    backtest_cfg = cfg.get("backtest", {})
    sku_ids, store_ids = backtest_cfg.get("sku_ids"), backtest_cfg.get("store_ids")
    if sku_ids:
        development = development.loc[development.item_id.astype(str).isin({str(x) for x in sku_ids})]
    if store_ids:
        development = development.loc[development.store_id.astype(str).isin({str(x) for x in store_ids})]
    for fold in development_splits(cfg):
        expected_parts.append(construct_inputs(development, fold).future_known[FUTURE_KEYS])
    expected = pd.concat(expected_parts, ignore_index=True).sort_values(FUTURE_KEYS).reset_index(drop=True)

    predictions: dict[str, pd.DataFrame] = {}
    metrics: list[pd.DataFrame] = []
    metadata: dict = {}
    destination = None if output_dir is None else Path(output_dir)
    for label, factory in factories.items():
        model_output = None if destination is None else destination / label
        forecast, score, meta = run_backtest(panel, cfg, factory, output_dir=model_output)
        if list(forecast.columns) != PREDICTION_COLUMNS:
            raise ValueError(f"{label} did not return the standardized prediction schema")
        keys = _comparable_grid(forecast[FUTURE_KEYS])
        expected_keys = _comparable_grid(expected)
        if not keys.equals(expected_keys):
            raise ValueError(f"{label} forecast grid differs from the shared cohort/origins/dates")
        if forecast.model.nunique() != 1:
            raise ValueError(f"{label} returned more than one model identity")
        actual_name = str(forecast.model.iloc[0])
        if actual_name in predictions:
            raise ValueError(f"Duplicate model identity {actual_name!r} in suite")
        predictions[actual_name] = forecast
        metrics.append(score)
        metadata[actual_name] = {**meta, "suite_label": label,
                                 "expected_forecast_rows": len(expected),
                                 "forecast_grid_match": True}

    combined_metrics = pd.concat(metrics, ignore_index=True)
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        pd.concat(predictions.values(), ignore_index=True).to_parquet(
            destination / "all_model_predictions.parquet", index=False)
        combined_metrics.to_csv(destination / "all_model_metrics.csv", index=False)
        (destination / "suite_metadata.json").write_text(json.dumps({
            "model_names": list(predictions),
            "expected_forecast_rows_per_model": len(expected),
            "common_origins": [fold.origin for fold in development_splits(cfg)],
            "common_series_count": int(expected.series_id.nunique()),
            "horizon": int(cfg["forecast"]["horizon"]),
            "model_runs": metadata,
        }, indent=2))
    return predictions, combined_metrics, metadata


def _comparable_grid(keys: pd.DataFrame) -> pd.DataFrame:
    """Normalize identifier/date dtypes before exact forecast-grid comparison.

    Adapters may return ``series_id`` as a categorical or an object/string
    column. Those representations can contain identical identifiers while
    ``DataFrame.equals`` reports them as unequal due only to dtype differences.
    """
    comparable = keys[FUTURE_KEYS].copy()
    comparable["series_id"] = comparable["series_id"].astype("string")
    comparable["target_date"] = pd.to_datetime(comparable["target_date"])
    return comparable.sort_values(FUTURE_KEYS).reset_index(drop=True)
