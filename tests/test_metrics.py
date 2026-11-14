import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_pinball_loss
from src.evaluation.metrics import (
    add_mase_scales, compute_metric_tables, mae, mase, mase_scale, pinball_loss,
    rmse, save_metric_tables, score_predictions, smape, wape,
)


def test_hand_calculated_metrics():
    actual, prediction = [0, 2, 4], [1, 2, 2]
    assert mae(actual, prediction) == 1
    assert rmse(actual, prediction) == pytest.approx(np.sqrt(5 / 3))
    assert wape(actual, prediction) == 0.5
    assert pinball_loss(actual, prediction, 0.5) == 0.5
    assert pinball_loss(actual, prediction, 0.1) == pytest.approx(1.1 / 3)
    assert pinball_loss(actual, prediction, 0.9) == pytest.approx(1.9 / 3)


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_pinball_against_sklearn(q):
    actual, prediction = np.array([0, 3, 1, 9]), np.array([1, 2, 2, 5])
    assert pinball_loss(actual, prediction, q) == mean_pinball_loss(actual, prediction, alpha=q)
    assert mae(actual, prediction) == mean_absolute_error(actual, prediction)
    assert rmse(actual, prediction) == root_mean_squared_error(actual, prediction)


@pytest.mark.parametrize("metric", [mae, rmse, wape, lambda y, p: pinball_loss(y, p, 0.5)])
def test_missing_observations_explicit(metric):
    assert metric([1, np.nan, 3], [1, np.nan, 3]) == 0
    assert np.isnan(metric([np.nan], [2]))
    with pytest.raises(ValueError, match="finite prediction"):
        metric([1, 2], [1, np.nan])


def test_zero_denominator_wape_is_undefined():
    assert np.isnan(wape([0, 0], [0, 0]))
    assert np.isnan(wape([0, 0], [1, 1]))


@pytest.mark.parametrize("q", [0, 1, -0.1, 1.1])
def test_invalid_quantile_rejected(q):
    with pytest.raises(ValueError): pinball_loss([1], [1], q)


def test_invalid_metric_shapes_and_actuals():
    with pytest.raises(ValueError): mae([1, 2], [1])
    with pytest.raises(ValueError): mae([np.inf], [1])


def test_smape_hand_calculation_missing_values_and_both_zero():
    assert smape([0, 2, 4], [1, 2, 2]) == pytest.approx(800 / 9)
    assert smape([0, np.nan, 2], [0, 999, 1]) == pytest.approx(100 / 3)
    assert np.isnan(smape([np.nan], [0]))
    with pytest.raises(ValueError, match="finite prediction"):
        smape([1], [np.nan])


def test_mase_uses_training_only_seasonal_scale_and_zero_scale_is_undefined():
    assert mase_scale([1, 2, 1, 4], seasonal_period=2) == pytest.approx(1.0)
    assert mase([3, 5], [1, 5], [1, 2, 1, 4], seasonal_period=2) == pytest.approx(1.0)
    assert np.isnan(mase([1], [2], [3, 3, 3], seasonal_period=1))
    assert np.isnan(mase_scale([1, 2], seasonal_period=2))


def test_mase_training_scale_ignores_rows_after_forecast_origin():
    scored = pd.DataFrame({"series_id": ["A", "A"], "origin": [5, 5],
                           "actual": [6.0, 7.0], "y_pred": [5.0, 7.0]})
    history = pd.DataFrame({"series_id": ["A"] * 7, "day_num": range(1, 8),
                            "units_sold": [1., 2., 3., 4., 5., 900., 1200.]})
    result = add_mase_scales(scored, history, seasonal_period=2)
    # Origin 5 permits only days 1..5: lag-2 errors are [2, 2, 2], scale=2.
    assert result.mase_scale.tolist() == [2.0, 2.0]
    assert score_predictions(result)["mase"] == pytest.approx(0.25)


def test_probabilistic_metrics_and_wis_hand_calculation():
    scored = pd.DataFrame({"actual": [0., 2., 4.], "y_pred": [1., 2., 3.],
                           "q10": [0., 1., 2.], "q50": [1., 2., 3.], "q90": [2., 3., 6.]})
    metrics = score_predictions(scored)
    assert metrics["pinball_q10"] == pytest.approx(0.1)
    assert metrics["pinball_q50"] == pytest.approx(1 / 3)
    assert metrics["pinball_q90"] == pytest.approx(1 / 6)
    assert metrics["coverage_80"] == 1.0
    assert metrics["mean_interval_width_80"] == pytest.approx(8 / 3)
    assert metrics["wis_80"] == pytest.approx(0.4)


def test_point_forecasts_have_no_probabilistic_scores_and_partial_quantiles_error():
    point = pd.DataFrame({"actual": [1., 2.], "y_pred": [1., 2.],
                          "q10": [np.nan, np.nan], "q50": [np.nan, np.nan], "q90": [np.nan, np.nan]})
    metrics = score_predictions(point)
    for key in ["pinball_q10", "pinball_q50", "pinball_q90", "coverage_80",
                 "mean_interval_width_80", "wis_80"]:
        assert np.isnan(metrics[key])
    point.loc[0, "q10"] = 0.5
    with pytest.raises(ValueError, match="all three finite quantiles or none"):
        score_predictions(point)


def test_metric_aggregation_dimensions_and_csv_export(tmp_path):
    scored = pd.DataFrame({
        "model": ["m1"] * 4, "series_id": ["A", "A", "B", "B"],
        "origin": [10, 10, 10, 10], "horizon": [1, 2, 1, 2],
        "actual": [1., 2., 3., 4.], "y_pred": [1., 1., 2., 4.],
        "q10": [np.nan] * 4, "q50": [np.nan] * 4, "q90": [np.nan] * 4,
        "mase_scale": [1.] * 4,
    })
    identity = pd.DataFrame({"series_id": ["A", "B"], "item_id": ["I1", "I2"],
                             "store_id": ["S1", "S2"]})
    tables = compute_metric_tables(scored, series_metadata=identity,
                                   demand_segments={"A": "low", "B": "high"})
    assert set(tables) == {"overall", "sku_store", "store", "origin", "horizon", "demand_segment"}
    assert len(tables["overall"]) == 1
    assert len(tables["sku_store"]) == 2
    assert len(tables["store"]) == 2
    assert len(tables["origin"]) == 1
    assert len(tables["horizon"]) == 2
    assert len(tables["demand_segment"]) == 2
    paths = save_metric_tables(tables, tmp_path, file_format="csv")
    assert len(paths) == 6
    assert pd.read_csv(tmp_path / "metrics_overall.csv").iloc[0].mae == pytest.approx(0.5)
