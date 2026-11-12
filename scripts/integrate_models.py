"""CLI runner for multi-model integration and comparative benchmarking.

Integrates and benchmarks models across all four paradigms:
- Seasonal Naive Baselines
- Kalman State-Space Variants
- LightGBM Gradient Boosted Decision Trees
- DeepAR Autoregressive Probabilistic RNNs

Examples:
    python -m scripts.integrate_models
    python -m scripts.integrate_models --config experiment.yaml --output-dir reports
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import data_path, load_config
from src.data.preprocess import load_panel
from src.evaluation.integration import (
    load_all_model_benchmarks,
    generate_integration_summary_markdown,
    verify_consistent_evaluation_inputs,
)
from src.evaluation.splits_validation import validate_chronological_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_integration_pipeline(
    cfg: dict,
    reports_dir: Path | str = "reports",
    processed_dir: Optional[Path | str] = None,
) -> tuple[pd.DataFrame, dict]:
    """Execute complete model integration and verification pipeline."""
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    proc_path = Path(processed_dir) if processed_dir else data_path(cfg, "processed")

    logger.info("1. Validating chronological splits and holdout isolation...")
    panel = load_panel(cfg, cutoff=cfg["forecast"]["final_origin"])
    split_validation = validate_chronological_splits(cfg, panel=panel)
    logger.info(
        f"Chronological splits validated: {split_validation['num_folds']} folds, "
        f"origins {split_validation['origins']}, holdout reserved (d_{split_validation['holdout_origin']} + 1 .. d_1941)"
    )

    logger.info("2. Discovering and verifying model predictions from disk...")
    # Load available parquet predictions to verify consistency
    predictions_map = {}

    # Check LightGBM
    lgb_parquet = proc_path / "lightgbm" / "predictions.parquet"
    if lgb_parquet.exists():
        logger.info(f"Loading LightGBM predictions from {lgb_parquet}")
        predictions_map["lightgbm"] = pd.read_parquet(lgb_parquet)

    # Check Seasonal Naive
    for naive_var in ["last_value", "weekly_seasonal"]:
        naive_pq = proc_path / "seasonal_naive" / naive_var / "predictions.parquet"
        if naive_pq.exists():
            model_key = f"naive_{naive_var}"
            logger.info(f"Loading Seasonal Naive ({naive_var}) from {naive_pq}")
            predictions_map[model_key] = pd.read_parquet(naive_pq)

    # Check Kalman
    kalman_dir = proc_path / "kalman_forecaster"
    if kalman_dir.exists():
        for var_dir in kalman_dir.iterdir():
            if var_dir.is_dir() and (var_dir / "predictions.parquet").exists():
                model_key = f"kalman_{var_dir.name}"
                logger.info(f"Loading Kalman ({var_dir.name}) from {var_dir / 'predictions.parquet'}")
                predictions_map[model_key] = pd.read_parquet(var_dir / "predictions.parquet")

    # Check DeepAR
    deepar_pq = proc_path / "deepar" / "predictions.parquet"
    if deepar_pq.exists():
        logger.info(f"Loading DeepAR predictions from {deepar_pq}")
        predictions_map["deepar"] = pd.read_parquet(deepar_pq)

    if predictions_map:
        logger.info(f"Verifying input consistency across {len(predictions_map)} discovered model outputs...")
        consistency_report = verify_consistent_evaluation_inputs(predictions_map, cfg, panel=panel)
        logger.info(f"Consistency verification passed: {consistency_report['status']}")
    else:
        logger.info("No saved Parquet predictions found on disk; proceeding to benchmark compilation from reports.")
        consistency_report = {"status": "BENCHMARK_ONLY"}

    logger.info("3. Compiling multi-model benchmark comparison across all 4 paradigms...")
    benchmarks_df = load_all_model_benchmarks(reports_path, data_dir=proc_path)

    # Save integrated artifacts
    out_csv = reports_path / "integrated_model_comparison.csv"
    benchmarks_df.to_csv(out_csv, index=False)
    logger.info(f"Saved integrated benchmark comparison to {out_csv}")

    # Generate markdown report
    summary_md = generate_integration_summary_markdown(benchmarks_df)
    out_md = reports_path / "integrated_model_summary.md"
    out_md.write_text(summary_md)
    logger.info(f"Saved integrated summary markdown to {out_md}")

    # Build origin breakdown table if report files exist
    origin_dfs = []
    kalman_orig = reports_path / "kalman_variant_by_origin.csv"
    if kalman_orig.exists():
        origin_dfs.append(pd.read_csv(kalman_orig))
    lgb_orig = reports_path / "m9_lightgbm_comparison.csv"
    if lgb_orig.exists():
        df_lgb = pd.read_csv(lgb_orig)
        fold_lgb = df_lgb[df_lgb["scope"] == "fold_pooled"].copy()
        if not fold_lgb.empty:
            origin_dfs.append(fold_lgb)
    naive_orig = reports_path / "m5_baseline_comparison.csv"
    if naive_orig.exists():
        df_naive = pd.read_csv(naive_orig)
        fold_naive = df_naive[df_naive["scope"] == "fold_pooled"].copy()
        if not fold_naive.empty:
            origin_dfs.append(fold_naive)

    if origin_dfs:
        common_cols = ["model", "origin", "mae", "rmse", "wape"]
        cleaned_origin_dfs = []
        for d in origin_dfs:
            c = [col for col in common_cols if col in d.columns]
            cleaned_origin_dfs.append(d[c])
        combined_origin = pd.concat(cleaned_origin_dfs, ignore_index=True).sort_values(["origin", "wape"])
        combined_origin.to_csv(reports_path / "integrated_model_by_origin.csv", index=False)
        logger.info(f"Saved integrated origin breakdown to {reports_path / 'integrated_model_by_origin.csv'}")

    # Build volume band breakdown table if available
    kalman_vol = reports_path / "kalman_variant_by_volume_band.csv"
    if kalman_vol.exists():
        df_vol = pd.read_csv(kalman_vol)
        df_vol.to_csv(reports_path / "integrated_model_by_volume.csv", index=False)
        logger.info(f"Saved integrated volume breakdown to {reports_path / 'integrated_model_by_volume.csv'}")

    summary_metadata = {
        "title": "Phase 1 Multi-Model Pipeline Integration",
        "split_validation": split_validation,
        "consistency_report": consistency_report,
        "models_count": len(benchmarks_df),
        "best_point_accuracy_model": benchmarks_df.iloc[0]["model"] if not benchmarks_df.empty else None,
        "best_point_accuracy_wape": float(benchmarks_df.iloc[0]["wape"]) if not benchmarks_df.empty else None,
        "generated_artifacts": [
            str(out_csv),
            str(out_md),
            str(reports_path / "integrated_model_by_origin.csv"),
            str(reports_path / "integrated_model_by_volume.csv"),
        ],
    }
    def json_serializer(obj: Any) -> Any:
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, list)):
            return [json_serializer(i) for i in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return str(obj)

    (reports_path / "integrated_model_summary.json").write_text(
        json.dumps(summary_metadata, indent=2, default=json_serializer)
    )
    logger.info(f"Saved integration metadata to {reports_path / 'integrated_model_summary.json'}")

    return benchmarks_df, summary_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiment.yaml", help="Path to experiment configuration YAML")
    parser.add_argument("--reports-dir", default="reports", help="Directory where reports are stored")
    parser.add_argument("--processed-dir", default=None, help="Directory where processed data is stored")
    args = parser.parse_args()

    cfg = load_config(args.config)
    benchmarks, meta = run_integration_pipeline(
        cfg=cfg,
        reports_dir=args.reports_dir,
        processed_dir=args.processed_dir,
    )

    print("\n" + "=" * 80)
    print("MULTI-MODEL INTEGRATED PERFORMANCE SUMMARY")
    print("=" * 80)
    print(benchmarks.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
