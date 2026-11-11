import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_pinball_loss
from src.evaluation.metrics import mae, rmse, wape, pinball_loss


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
