import pandas as pd

from src.evaluation.backtesting import DummyForecaster, FUTURE_KEYS
from src.evaluation.model_suite import run_model_suite


def test_model_suite_enforces_common_grid_and_persists_outputs(tiny_panel, tmp_path):
    panel, cfg = tiny_panel
    predictions, metrics, metadata = run_model_suite(
        panel,
        cfg,
        {"zero": lambda: DummyForecaster(0),
         "one": lambda: DummyForecaster(1)},
        output_dir=tmp_path,
    )
    assert set(predictions) == {"dummy_zero", "dummy_constant"}
    reference = predictions["dummy_zero"][FUTURE_KEYS].sort_values(FUTURE_KEYS).reset_index(drop=True)
    other = predictions["dummy_constant"][FUTURE_KEYS].sort_values(FUTURE_KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(reference, other)
    assert all(details["forecast_grid_match"] for details in metadata.values())
    assert metrics.model.nunique() == 2
    combined = pd.read_parquet(tmp_path / "all_model_predictions.parquet")
    assert combined.model.nunique() == 2
    assert (tmp_path / "all_model_metrics.csv").is_file()
    assert (tmp_path / "suite_metadata.json").is_file()
