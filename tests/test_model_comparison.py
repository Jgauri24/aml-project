import numpy as np
import pandas as pd
import pytest

from src.evaluation.model_comparison import (
    bootstrap_performance_differences,
    compare_models,
    diebold_mariano_test,
    paired_loss_comparisons,
    rank_models,
    save_comparison_results,
)


def forecast_frame(n_origins=10, n_series=3, horizon=2):
    rows = []
    for origin in range(20, 20 + n_origins * horizon, horizon):
        for series_idx in range(n_series):
            for h in range(1, horizon + 1):
                actual = float(series_idx + h + origin / 100)
                rows.extend([
                    {"model": "A", "series_id": f"S{series_idx}", "origin": origin,
                     "horizon": h, "target_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=origin + h),
                     "actual": actual, "y_pred": actual + 1., "q10": max(0., actual),
                     "q50": actual + 1., "q90": actual + 2.},
                    {"model": "B", "series_id": f"S{series_idx}", "origin": origin,
                     "horizon": h, "target_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=origin + h),
                     "actual": actual, "y_pred": actual + 2., "q10": max(0., actual + 1.),
                     "q50": actual + 2., "q90": actual + 3.},
                ])
    return pd.DataFrame(rows)


def test_paired_loss_differences_are_aligned_and_hand_verifiable():
    result = paired_loss_comparisons(forecast_frame(n_origins=1, n_series=2), "A", "B")
    absolute = result.loc[result.loss.eq("absolute_error")].iloc[0]
    squared = result.loc[result.loss.eq("squared_error")].iloc[0]
    assert absolute.mean_difference_a_minus_b == pytest.approx(-1.)
    assert absolute.median_difference == pytest.approx(-1.)
    assert squared.mean_difference_a_minus_b == pytest.approx(-3.)
    assert absolute.n_paired_rows == 4


def test_dm_aggregates_by_origin_and_refuses_too_few_origins():
    data = forecast_frame(n_origins=4, n_series=20)
    result = diebold_mariano_test(data, "A", "B", horizon=2, origin_step=2)
    assert result["n_origins"] == 4
    assert result["n_series"] == 20
    assert result["status"] == "insufficient_origins"
    assert np.isnan(result["p_value"])
    assert result["hac_lag"] == 0


def test_dm_hac_and_small_sample_test_run_when_origin_count_is_sufficient():
    result = diebold_mariano_test(forecast_frame(), "A", "B", horizon=2, origin_step=1)
    assert result["status"] == "ok"
    assert result["hac_lag"] == 1
    assert result["dm_statistic"] < 0
    assert result["p_value"] < .05


def test_paired_rows_reject_different_actuals():
    data = forecast_frame(n_origins=1, n_series=1)
    data.loc[data.model.eq("B"), "actual"] += 1
    with pytest.raises(ValueError, match="different/missing evaluation actuals"):
        paired_loss_comparisons(data, "A", "B")


def test_cluster_bootstrap_is_reproducible_and_reports_zero_in_interval():
    data = forecast_frame(n_origins=3, n_series=3)
    data.loc[data.model.eq("B"), "y_pred"] = data.loc[data.model.eq("A"), "y_pred"].to_numpy()
    one = bootstrap_performance_differences(data, "A", "B", metrics=("mae",), n_bootstrap=100, seed=7)
    two = bootstrap_performance_differences(data, "A", "B", metrics=("mae",), n_bootstrap=100, seed=7)
    pd.testing.assert_frame_equal(one, two)
    assert one.iloc[0].difference_a_minus_b == 0
    assert one.iloc[0].ci_lower == 0 and one.iloc[0].ci_upper == 0
    assert one.iloc[0].zero_in_ci


def test_cluster_bootstrap_scores_aligned_probabilistic_quantiles():
    result = bootstrap_performance_differences(
        forecast_frame(n_origins=3, n_series=3), "A", "B",
        metrics=("pinball_q10", "pinball_q50", "pinball_q90", "wis_80"),
        n_bootstrap=50, seed=9,
    )
    assert result.status.eq("ok").all()
    assert result.difference_a_minus_b.lt(0).all()
    assert result.ci_upper.lt(0).all()


def test_ranking_is_separate_by_horizon_segment_probability_and_runtime(tmp_path):
    data = forecast_frame(n_origins=3, n_series=3)
    rankings = rank_models(
        data,
        demand_segments={"S0": "low", "S1": "high", "S2": "high"},
        execution=pd.DataFrame({"model": ["A", "B"], "fit_seconds": [2., 1.],
                                "predict_seconds": [1., 2.]}),
    )
    assert set(rankings) == {"overall_accuracy", "horizon_performance", "demand_segment_performance",
                             "probabilistic_performance", "computational_efficiency"}
    assert rankings["overall_accuracy"].query("metric == 'mae'").sort_values("rank").iloc[0].model == "A"
    assert len(rankings["horizon_performance"]) == 8
    assert len(rankings["demand_segment_performance"]) == 8
    assert len(rankings["probabilistic_performance"]) > 0
    assert rankings["computational_efficiency"].query("metric == 'fit_seconds'").sort_values("rank").iloc[0].model == "B"
    paths = save_comparison_results(rankings, tmp_path)
    assert len(paths) == 5
    assert (tmp_path / "overall_accuracy.csv").exists()


def test_compare_models_holm_adjustment_and_exports_pair_tables():
    data = forecast_frame()
    results = compare_models(data, [("A", "B")], horizon=2, origin_step=2,
                             n_bootstrap=20, seed=12)
    assert set(results) == {"paired_comparisons", "dm_tests", "bootstrap"}
    assert results["dm_tests"].p_adjusted_holm.notna().any()
    assert results["bootstrap"].query("metric == 'mae'").iloc[0].ci_upper < 0
