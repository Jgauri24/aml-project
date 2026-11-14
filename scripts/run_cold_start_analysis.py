#!/usr/bin/env python
"""Runner script for Member 4 — Cold-Start and Sparse Demand Analysis.

Usage
-----
    python scripts/run_cold_start_analysis.py [--output-dir reports/]

This script orchestrates the full Member 4 analysis pipeline:
  1. Demand-pattern segmentation of the 240-series cohort.
  2. Cold-start window degradation experiment.
  3. Stockout censoring bias analysis.
  4. Local vs global model comparison by demand segment.
  5. Failure case documentation.
  6. Figure generation and report export.

All results are written to ``reports/`` by default.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path regardless of invocation directory
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis.cold_start_analysis import (
    _build_series_segments,
    analyze_censoring_bias,
    compare_local_vs_global_by_segment,
    document_failure_cases,
    evaluate_short_window_performance,
    generate_cold_start_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Member 4 cold-start and sparse demand analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        default=str(_ROOT / "reports"),
        help="Directory to write reports and figures (default: reports/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all analysis steps but do not write any files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Member 4 — Cold-Start & Sparse Demand Analysis")
    logger.info("Phase 2 | AML Project")
    logger.info("=" * 60)

    t0 = time.perf_counter()

    # Quick sanity check: verify input data files exist
    reports = Path(args.output_dir)
    expected_inputs = [
        reports / "integrated_model_comparison.csv",
        reports / "integrated_model_by_volume.csv",
    ]
    for p in expected_inputs:
        if p.exists():
            logger.info("  ✓ Found: %s", p.name)
        else:
            logger.warning("  ✗ Missing: %s (will use fallback data)", p.name)

    if args.dry_run:
        logger.info("--- DRY RUN: validating analysis steps only ---")
        seg_df = _build_series_segments()
        logger.info("Segmentation: %d series classified", len(seg_df))
        window_df = evaluate_short_window_performance()
        logger.info("Window analysis: %d rows", len(window_df))
        censoring = analyze_censoring_bias()
        logger.info("Censoring bands: %s", censoring["volume_bands"])
        seg_comp = compare_local_vs_global_by_segment()
        logger.info("Segment comparison: %d rows", len(seg_comp))
        failures = document_failure_cases()
        logger.info("Failure cases: %d identified", len(failures))
        logger.info("Dry run complete — no files written.")
        return 0

    try:
        results = generate_cold_start_report(output_dir=args.output_dir)
    except Exception:
        logger.exception("Analysis failed")
        return 1

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("Analysis complete in %.1f s", elapsed)
    logger.info("Outputs written to: %s", args.output_dir)
    logger.info("")
    logger.info("Key results:")
    logger.info("  Cohort: %d series, %d folds, %d-day horizon",
                results["cohort"]["n_series"],
                results["cohort"]["n_folds"],
                results["cohort"]["horizon_days"])
    logger.info("  Demand segments: %s",
                dict(results["segmentation"]["counts"]))
    logger.info("  Best overall model: %s (MAE=%.4f)",
                results["best_model_overall"]["name"],
                results["best_model_overall"]["mae"])
    logger.info("  Cold-start robust: %s",
                results["cold_start_window_analysis"]["most_robust_model"])
    logger.info("  Cold-start sensitive: %s",
                results["cold_start_window_analysis"]["most_sensitive_model"])
    logger.info("  Failure cases: %d identified",
                results["failure_cases"]["n_identified"])
    logger.info("")
    logger.info("Files generated:")
    logger.info("  reports/cold_start_analysis.json")
    logger.info("  reports/cold_start_analysis.md")
    logger.info("  reports/cold_start_segment_comparison.csv")
    logger.info("  reports/cold_start_window_performance.csv")
    logger.info("  reports/cold_start_failure_cases.csv")
    logger.info("  reports/cold_start_series_segments.csv")
    for name, path in results.get("figures", {}).items():
        logger.info("  %s", path)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
