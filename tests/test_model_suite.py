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


def test_model_suite_accepts_equivalent_string_and_categorical_series_ids(tiny_panel):
    panel, cfg = tiny_panel

    class StringIdForecaster(DummyForecaster):
        def predict(self, inputs):
            forecasts = super().predict(inputs)
            forecasts["series_id"] = forecasts["series_id"].astype(str)
            return forecasts

    predictions, _, metadata = run_model_suite(
        panel,
        cfg,
        {"categorical": lambda: DummyForecaster(0),
         "string": lambda: StringIdForecaster(1)},
    )
    assert all(details["forecast_grid_match"] for details in metadata.values())
    categorical = predictions["dummy_zero"][FUTURE_KEYS].copy()
    string = predictions["dummy_constant"][FUTURE_KEYS].copy()
    categorical["series_id"] = categorical["series_id"].astype(str)
    string["series_id"] = string["series_id"].astype(str)
    pd.testing.assert_frame_equal(
        categorical.sort_values(FUTURE_KEYS).reset_index(drop=True),
        string.sort_values(FUTURE_KEYS).reset_index(drop=True),
    )
