"""Analytical examples and leakage/grid guards for Milestone 15."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.analysis.uncertainty_analysis import (
    analyze_uncertainty, diagnostic_metrics, history_labels, prepare_scored,
)
from src.evaluation.backtesting import DummyForecaster, run_backtest
from src.evaluation.integration import load_all_model_benchmarks


def forecasts(panel, cfg):
    prediction, *_ = run_backtest(panel, cfg, lambda: DummyForecaster(1))
    prediction['model'] = 'kalman_test'
    prediction[['q10', 'q50', 'q90']] = [0., 1., 3.]
    other = prediction.copy()
    other['model'] = 'deepar_test'
    return pd.concat([prediction, other], ignore_index=True)


def test_analytical_coverage_tails_pinball_and_discrete_bounds():
    frame = pd.DataFrame(dict(actual=[0., 1., 2., 5., np.nan], y_pred=[1.] * 5,
                              q10=[1.] * 5, q50=[2.] * 5, q90=[3.] * 5))
    scores = diagnostic_metrics(frame)
    assert scores['n_observed'] == scores['n_intervals'] == 4
    assert scores['coverage_80'] == .5
    assert scores['lower_miss_rate'] == scores['upper_miss_rate'] == .25
    assert scores['coverage_80'] + scores['lower_miss_rate'] + scores['upper_miss_rate'] == 1
    assert scores['mean_interval_width_80'] == 2
    assert scores['q10_cdf_lt'] == .25
    assert scores['q10_cdf_le'] == .5
    assert scores['pinball_q50'] == .75


def test_point_models_and_unobserved_groups_have_no_intervals():
    point = pd.DataFrame(dict(actual=[0., np.nan], y_pred=[1., 1.], q10=[np.nan]*2,
                              q50=[np.nan]*2, q90=[np.nan]*2))
    scores = diagnostic_metrics(point)
    assert scores['n_intervals'] == 0
    assert np.isnan(scores['coverage_80'])
    point[['q10', 'q50', 'q90']] = [0., 1., 2.]
    point['actual'] = np.nan
    assert diagnostic_metrics(point)['n_intervals'] == 0


def test_common_actuals_and_explicit_pairwise_ties(tiny_panel):
    panel, cfg = tiny_panel
    prediction = forecasts(panel, cfg)
    prediction['actual'] = 999999.  # Untrusted saved actuals never enter scoring.
    panel.loc[(panel.series_id == panel.series_id.iloc[0]) & panel.day_num.eq(7), 'units_sold'] = np.nan
    tables = analyze_uncertainty(prediction, panel, cfg)
    assert tables['overall'].n_observed.nunique() == 1
    assert tables['overall'].n_observed.iloc[0] == 17
    assert tables['paired_series'].winner.eq('tie').all()
    assert tables['overall'].mae.max() < 50
    assert set(tables['by_week'].horizon_week) == {1}


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'holdout', 'crossed', 'partial_quantiles', 'date'])
def test_reject_invalid_forecasts(tiny_panel, damage):
    panel, cfg = tiny_panel
    p = forecasts(panel, cfg)
    if damage == 'missing':
        p = p.iloc[1:]
    elif damage == 'duplicate':
        p = pd.concat([p, p.iloc[:1]])
    elif damage == 'holdout':
        p.loc[0, 'origin'] = cfg['forecast']['final_origin']
    elif damage == 'crossed':
        p.loc[0, 'q10'] = 10
    elif damage == 'partial_quantiles':
        p.loc[0, 'q90'] = np.nan
    elif damage == 'date':
        p.loc[0, 'target_date'] += pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        prepare_scored(p, panel, cfg)


def test_labels_ignore_future_and_respect_sliding_history(tiny_panel):
    panel, cfg = tiny_panel
    first = history_labels(panel, cfg).query('origin == 6').reset_index(drop=True)
    changed = panel.copy()
    changed.loc[changed.day_num.gt(6), 'units_sold'] = 99999
    second = history_labels(changed, cfg).query('origin == 6').reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)
    sliding = deepcopy(cfg)
    sliding['backtest'].update(window_mode='sliding', training_window_days=2)
    labels = history_labels(panel, sliding)
    assert labels.history_observed.eq(2).all()
    assert labels.trailing_zero_days.max() <= 2


def test_deepar_summary_does_not_invent_calibration(tmp_path):
    pd.DataFrame([dict(experiment='test', mae=1., rmse=1., wape=.5,
                       pinball_q10=.1, pinball_q50=.2, pinball_q90=.3)]).to_csv(
                           tmp_path / 'deepar_experiments_comparison.csv', index=False)
    table = load_all_model_benchmarks(tmp_path, data_dir=tmp_path)
    assert table.empirical_coverage.isna().all()
    assert table.mean_interval_width.isna().all()


def test_runner_writes_reproducible_artifacts(tiny_project, tmp_path):
    import json
    from src.config import data_path
    from src.data.preprocess import run_preprocessing
    from scripts.run_uncertainty_analysis import run_analysis

    cfg, _ = tiny_project
    panel, _ = run_preprocessing(cfg)
    path = data_path(cfg, 'processed') / 'saved_predictions.parquet'
    forecasts(panel, cfg).to_parquet(path, index=False)
    output = tmp_path / 'diagnostics'
    tables, metadata = run_analysis(cfg, [path], output)
    assert (output / 'calibration_by_week.png').is_file()
    assert (output / 'coverage_by_volume.png').is_file()
    assert (output / 'report.md').is_file()
    assert json.loads((output / 'manifest.json').read_text())['final_holdout_evaluated'] is False
    assert len(metadata['sources'][0]['sha256']) == 64
    saved = pd.read_csv(output / 'overall.csv')
    np.testing.assert_allclose(saved.mae, tables['overall'].mae)


def test_deepar_summary_preserves_measured_calibration(tmp_path):
    pd.DataFrame([dict(experiment='test', mae=1., rmse=1., wape=.5,
                       pinball_q10=.1, pinball_q50=.2, pinball_q90=.3,
                       empirical_coverage=.75, mean_interval_width=4.)]).to_csv(
                           tmp_path / 'deepar_experiments_comparison.csv', index=False)
    table = load_all_model_benchmarks(tmp_path, data_dir=tmp_path)
    assert table.empirical_coverage.iloc[0] == .75
    assert table.mean_interval_width.iloc[0] == 4.
