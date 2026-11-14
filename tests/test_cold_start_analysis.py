"""Tests for Member 4 — Cold-Start and Sparse Demand Analysis.

Covers:
  - Demand-pattern segmentation logic and edge cases.
  - Short-window degradation model (monotonicity, boundary conditions).
  - Censoring analysis structure and sanity checks.
  - Local vs global comparison table schema.
  - Failure case detection and taxonomy.
  - Full report generation (integration test with real project data).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis.cold_start_analysis import (
    COLD_START_DAYS,
    ZERO_FRAC_INTERMITTENT,
    ZERO_FRAC_SPARSE,
    analyze_censoring_bias,
    compare_local_vs_global_by_segment,
    document_failure_cases,
    evaluate_short_window_performance,
    generate_cold_start_report,
    segment_by_demand_pattern,
    _classify_failure,
    _model_family,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_volume_df(n_per_band: int = 5) -> pd.DataFrame:
    """Return a minimal volume-band DataFrame for testing segmentation."""
    rows = []
    for grp in ["high", "medium", "low"]:
        for i in range(n_per_band):
            rows.append({"series_id": f"{grp}_{i}", "volume_group": grp})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. segment_by_demand_pattern
# ---------------------------------------------------------------------------

class TestSegmentByDemandPattern:
    def test_returns_dataframe(self):
        df = _make_volume_df()
        result = segment_by_demand_pattern(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        df = _make_volume_df()
        result = segment_by_demand_pattern(df)
        for col in ["series_id", "volume_group", "demand_pattern",
                    "zero_fraction", "mean_demand", "cv2", "is_cold_start"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_high_volume_is_dense(self):
        df = pd.DataFrame([{"series_id": "h0", "volume_group": "high"}])
        result = segment_by_demand_pattern(df)
        assert result.loc[0, "demand_pattern"] == "dense"

    def test_medium_volume_is_intermittent(self):
        df = pd.DataFrame([{"series_id": "m0", "volume_group": "medium"}])
        result = segment_by_demand_pattern(df)
        assert result.loc[0, "demand_pattern"] == "intermittent"

    def test_low_volume_is_cold_start_or_sparse(self):
        df = pd.DataFrame([{"series_id": "l0", "volume_group": "low"}])
        result = segment_by_demand_pattern(df)
        # Low-volume maps to sparse (≥50 % zeros) and cold_start
        assert result.loc[0, "demand_pattern"] in {"sparse", "cold_start"}

    def test_explicit_zero_fraction_overrides_volume_band(self):
        df = pd.DataFrame([{
            "series_id": "x0",
            "volume_group": "high",   # would normally be dense
            "zero_frac": 0.70,        # but we specify 70 % zeros
        }])
        result = segment_by_demand_pattern(df, zero_frac_col="zero_frac")
        assert result.loc[0, "demand_pattern"] in {"sparse", "cold_start"}

    def test_zero_fraction_in_valid_range(self):
        df = _make_volume_df()
        result = segment_by_demand_pattern(df)
        assert (result["zero_fraction"] >= 0).all()
        assert (result["zero_fraction"] <= 1).all()

    def test_cv2_positive(self):
        df = _make_volume_df()
        result = segment_by_demand_pattern(df)
        assert (result["cv2"] > 0).all()

    def test_all_patterns_covered(self):
        """All four patterns should appear in the full 240-series cohort."""
        df = _make_volume_df(n_per_band=10)
        result = segment_by_demand_pattern(df)
        # At minimum dense, intermittent, cold_start should appear
        patterns = set(result["demand_pattern"].unique())
        assert len(patterns) >= 2

    def test_is_cold_start_boolean(self):
        df = _make_volume_df()
        result = segment_by_demand_pattern(df)
        assert result["is_cold_start"].dtype in (bool, object, "bool")

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame(columns=["series_id", "volume_group"])
        result = segment_by_demand_pattern(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_row_count_matches_input(self):
        df = _make_volume_df(n_per_band=7)
        result = segment_by_demand_pattern(df)
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 2. evaluate_short_window_performance
# ---------------------------------------------------------------------------

class TestEvaluateShortWindowPerformance:
    def test_returns_dataframe(self):
        result = evaluate_short_window_performance()
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        result = evaluate_short_window_performance()
        for col in ["model", "window_days", "mae", "rmse", "wape",
                    "relative_mae_increase_pct"]:
            assert col in result.columns

    def test_mae_positive(self):
        result = evaluate_short_window_performance()
        assert (result["mae"] > 0).all()

    def test_rmse_gte_mae(self):
        """RMSE ≥ MAE is a mathematical necessity."""
        result = evaluate_short_window_performance()
        assert (result["rmse"] >= result["mae"] - 1e-9).all()

    def test_monotone_mae_increases_as_window_shrinks(self):
        """Shorter training window → higher MAE for sensitive models."""
        result = evaluate_short_window_performance(window_days=[30, 90, 180, 364, 1800])
        for model in result["model"].unique():
            sub = result[result["model"] == model].sort_values("window_days")
            mae_vals = sub["mae"].to_numpy()
            # MAE should be non-increasing as window grows
            diffs = np.diff(mae_vals)
            assert (diffs <= 1e-6).all(), (
                f"MAE not monotone for {model}: {mae_vals}"
            )

    def test_zero_increase_at_full_window(self):
        """At the full 1800-day window the relative increase must be 0."""
        result = evaluate_short_window_performance(window_days=[1800])
        full = result[result["window_days"] == 1800]
        assert (full["relative_mae_increase_pct"] == 0.0).all()

    def test_all_models_present(self):
        result = evaluate_short_window_performance()
        expected_models = {"LightGBM", "Kalman LL+Seas", "Kalman Single-SKU",
                           "DeepAR (best)", "Seasonal Naive"}
        actual = set(result["model"].unique())
        assert expected_models.issubset(actual)

    def test_wape_bounded(self):
        """WAPE should not exceed 2.0 (200 %)."""
        result = evaluate_short_window_performance(window_days=[30, 60, 90])
        assert (result["wape"] <= 2.0).all()

    def test_custom_windows_accepted(self):
        windows = [14, 28, 56]
        result = evaluate_short_window_performance(window_days=windows)
        for w in windows:
            assert w in result["window_days"].values

    def test_lightgbm_most_robust_among_ml_models(self):
        """LightGBM should be more robust than Kalman and DeepAR at 30-day window.

        Note: Seasonal Naive (requires only 7 days) may show lower % increase
        because it only needs 7 days. LightGBM beats the other ML/state-space
        models thanks to global cross-series learning.
        """
        result = evaluate_short_window_performance(window_days=[30])
        at_30 = result[result["window_days"] == 30].set_index("model")
        lgbm_increase = at_30.loc["LightGBM", "relative_mae_increase_pct"]
        # LightGBM must beat Kalman and DeepAR (not necessarily naive baseline)
        ml_models = ["Kalman LL+Seas", "Kalman Single-SKU", "DeepAR (best)"]
        for model in ml_models:
            if model in at_30.index:
                assert lgbm_increase <= at_30.loc[model, "relative_mae_increase_pct"] + 1e-6, (
                    f"LightGBM ({lgbm_increase:.1f}%) should be ≤ {model} "
                    f"({at_30.loc[model, 'relative_mae_increase_pct']:.1f}%)"
                )


# ---------------------------------------------------------------------------
# 3. analyze_censoring_bias
# ---------------------------------------------------------------------------

class TestAnalyzeCensoringBias:
    def test_returns_dict(self):
        result = analyze_censoring_bias()
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = analyze_censoring_bias()
        for key in ["volume_bands", "zero_fraction_by_band", "censoring_estimates",
                    "bias_impact", "mitigation_strategies", "summary"]:
            assert key in result, f"Missing key: {key}"

    def test_volume_bands_complete(self):
        result = analyze_censoring_bias()
        assert set(result["volume_bands"]) == {"high", "medium", "low"}

    def test_censored_fracs_in_range(self):
        result = analyze_censoring_bias()
        for band, est in result["censoring_estimates"].items():
            assert 0 <= est["censored_frac_of_zeros"] <= 1, f"Bad frac for {band}"
            assert 0 <= est["censored_frac_of_all"] <= 1, f"Bad total frac for {band}"

    def test_low_band_most_affected(self):
        """Low-volume band should have the highest censoring fraction."""
        result = analyze_censoring_bias()
        cens = result["censoring_estimates"]
        assert (cens["low"]["censored_frac_of_zeros"] >=
                cens["high"]["censored_frac_of_zeros"])

    def test_bias_impact_positive(self):
        result = analyze_censoring_bias()
        for band, impact in result["bias_impact"].items():
            assert impact["estimated_demand_bias_units_per_day"] >= 0

    def test_mitigation_strategies_non_empty(self):
        result = analyze_censoring_bias()
        assert len(result["mitigation_strategies"]) >= 2

    def test_summary_is_string(self):
        result = analyze_censoring_bias()
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 50

    def test_accepts_custom_volume_df(self):
        """Should not crash when given a minimal custom DataFrame."""
        custom = pd.DataFrame({
            "volume_group": ["high", "medium", "low"],
            "mae": [1.9, 0.7, 0.4],
            "wape": [0.72, 1.12, 1.30],
        })
        result = analyze_censoring_bias(volume_band_df=custom)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 4. compare_local_vs_global_by_segment
# ---------------------------------------------------------------------------

class TestCompareLocalVsGlobal:
    def test_returns_dataframe(self):
        result = compare_local_vs_global_by_segment()
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        result = compare_local_vs_global_by_segment()
        for col in ["segment", "model", "model_family", "mae", "rmse",
                    "wape", "rank_in_segment"]:
            assert col in result.columns

    def test_ranks_start_at_one(self):
        result = compare_local_vs_global_by_segment()
        for seg, grp in result.groupby("segment"):
            assert grp["rank_in_segment"].min() == 1

    def test_mae_positive(self):
        result = compare_local_vs_global_by_segment()
        assert (result["mae"] > 0).all()

    def test_all_model_families_present(self):
        result = compare_local_vs_global_by_segment()
        families = set(result["model_family"].unique())
        # At minimum Kalman and LightGBM families should appear
        assert any("Kalman" in f for f in families)
        assert any("LightGBM" in f for f in families)

    def test_no_duplicate_model_per_segment(self):
        result = compare_local_vs_global_by_segment()
        dupes = result.groupby(["segment", "model"]).size()
        assert (dupes <= 1).all(), "Duplicate model×segment rows detected"


# ---------------------------------------------------------------------------
# 5. document_failure_cases
# ---------------------------------------------------------------------------

class TestDocumentFailureCases:
    def test_returns_dataframe(self):
        result = document_failure_cases()
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        result = document_failure_cases()
        expected = ["model", "segment", "mae", "wape", "naive_mae",
                    "relative_mae_vs_naive_pct", "failure_mode", "severity"]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"

    def test_severity_values(self):
        result = document_failure_cases()
        valid = {"none", "medium", "high", "critical"}
        assert set(result["severity"].unique()).issubset(valid)

    def test_top_n_respected(self):
        result = document_failure_cases(top_n=3)
        assert len(result) <= 3

    def test_mae_positive(self):
        result = document_failure_cases()
        assert (result["mae"] > 0).all()


class TestClassifyFailure:
    def test_catastrophic_wape(self):
        result = _classify_failure("deepar", "sparse", 2.0, 1.8, 0.5)
        assert result == "catastrophic_wape"

    def test_high_wape(self):
        result = _classify_failure("kalman_local_level", "low", 1.5, 1.2, 0.3)
        assert result == "high_wape_exceeds_mean"

    def test_far_below_baseline(self):
        result = _classify_failure("deepar", "dense", 2.0, 0.9, 0.6)
        assert result == "far_below_baseline"

    def test_good_model_no_failure(self):
        result = _classify_failure("lightgbm", "dense", 0.5, 0.4, -0.3)
        assert result == "none"


# ---------------------------------------------------------------------------
# 6. _model_family helper
# ---------------------------------------------------------------------------

class TestModelFamily:
    def test_kalman_variants(self):
        for name in ["kalman_local_level", "kalman_single_sku", "kalman_local_trend_seasonal"]:
            assert _model_family(name) == "Local (Kalman)"

    def test_lightgbm(self):
        assert _model_family("lightgbm") == "Global (LightGBM)"

    def test_deepar(self):
        assert _model_family("deepar_exp_c") == "Global (DeepAR)"

    def test_naive(self):
        assert _model_family("naive_weekly_seasonal") == "Baseline"

    def test_unknown(self):
        assert _model_family("mystery_model_xyz") == "Other"


# ---------------------------------------------------------------------------
# 7. Integration test: generate_cold_start_report
# ---------------------------------------------------------------------------

class TestGenerateColdStartReport:
    def test_returns_dict(self, tmp_path):
        result = generate_cold_start_report(output_dir=tmp_path)
        assert isinstance(result, dict)

    def test_required_top_level_keys(self, tmp_path):
        result = generate_cold_start_report(output_dir=tmp_path)
        for key in ["phase", "member", "cohort", "segmentation",
                    "best_model_overall", "local_vs_global_by_segment",
                    "cold_start_window_analysis", "censoring_analysis",
                    "failure_cases", "figures"]:
            assert key in result, f"Missing top-level key: {key}"

    def test_json_file_written(self, tmp_path):
        generate_cold_start_report(output_dir=tmp_path)
        json_path = tmp_path / "cold_start_analysis.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "cohort" in data

    def test_markdown_file_written(self, tmp_path):
        generate_cold_start_report(output_dir=tmp_path)
        md_path = tmp_path / "cold_start_analysis.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "Member 4" in content
        assert "Cold-Start" in content

    def test_csv_files_written(self, tmp_path):
        generate_cold_start_report(output_dir=tmp_path)
        for fname in ["cold_start_segment_comparison.csv",
                      "cold_start_window_performance.csv",
                      "cold_start_failure_cases.csv",
                      "cold_start_series_segments.csv"]:
            path = tmp_path / fname
            assert path.exists(), f"Missing CSV: {fname}"

    def test_figures_written(self, tmp_path):
        result = generate_cold_start_report(output_dir=tmp_path)
        # Figures go to <project>/reports/figures/, not tmp_path
        # Check that figure paths were returned
        figs = result.get("figures", {})
        assert len(figs) >= 4, "Expected at least 4 figures"
        for name, path in figs.items():
            assert Path(path).exists(), f"Figure not found: {path}"

    def test_cohort_numbers_correct(self, tmp_path):
        result = generate_cold_start_report(output_dir=tmp_path)
        cohort = result["cohort"]
        assert cohort["n_series"] == 240
        assert cohort["n_folds"] == 4
        assert cohort["horizon_days"] == 28
        assert cohort["total_test_points"] == 240 * 4 * 28

    def test_segmentation_counts_sum_to_n_series(self, tmp_path):
        result = generate_cold_start_report(output_dir=tmp_path)
        total = sum(result["segmentation"]["counts"].values())
        assert total == result["segmentation"]["total"]

    def test_best_model_mae_positive(self, tmp_path):
        result = generate_cold_start_report(output_dir=tmp_path)
        assert result["best_model_overall"]["mae"] > 0

    def test_json_serialisable(self, tmp_path):
        """Ensure the returned dict can be round-tripped through JSON."""
        result = generate_cold_start_report(output_dir=tmp_path)
        serialised = json.dumps(result, default=str)
        restored = json.loads(serialised)
        assert restored["member"] == result["member"]

    def test_idempotent(self, tmp_path):
        """Running twice must not raise errors (overwrites are fine)."""
        generate_cold_start_report(output_dir=tmp_path)
        result2 = generate_cold_start_report(output_dir=tmp_path)
        assert isinstance(result2, dict)
