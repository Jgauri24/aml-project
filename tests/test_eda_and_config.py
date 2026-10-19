from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
import yaml
from src.config import load_config
from src.data.eda import analyze, longest_zero_run, autocorrelation
from src.data.select_skus import select_series
from src.data.load_data import load_sales


def test_zero_runs_and_acf():
    assert longest_zero_run([1, 0, 0, np.nan, 0, 0, 0, 1]) == 3
    assert longest_zero_run([1, 2]) == 0
    assert longest_zero_run([0, 0]) == 2
    assert np.isnan(autocorrelation([0, 0, 0], 1))
    assert autocorrelation([1, 2, 1, 2, 1, 2], 2) == pytest.approx(1)


def test_eda_does_not_see_future_targets_or_prices(tiny_panel, tiny_project):
    panel, cfg = tiny_panel
    _, raw = tiny_project
    selected, _ = select_series(load_sales(raw, 6), cfg)
    original_safe, original_stats, original_summaries = analyze(panel, selected, cfg)
    changed = panel.copy()
    changed.loc[changed.day_num.gt(6), ["units_sold", "sell_price"]] = 100000
    safe, stats, summaries = analyze(changed, selected, cfg)
    pd.testing.assert_frame_equal(original_safe, safe)
    pd.testing.assert_frame_equal(original_stats, stats)
    for name in summaries:
        pd.testing.assert_frame_equal(original_summaries[name], summaries[name])
    assert safe.day_num.max() == 6


@pytest.mark.parametrize("problem", ["selection", "eda", "overlap", "availability", "quantiles"])
def test_unsafe_config_rejected(tmp_path, problem):
    cfg = deepcopy(load_config())
    if problem == "selection": cfg["selection"]["cutoff_day"] = 1820
    if problem == "eda": cfg["eda"]["cutoff_day"] = 1802
    if problem == "overlap": cfg["forecast"]["development_origins"] = [1801, 1913]
    if problem == "availability": cfg["availability"]["future_prices"] = "actual"
    if problem == "quantiles": cfg["metrics"]["quantiles"] = [0.2]
    path = tmp_path / "unsafe.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError): load_config(path)
