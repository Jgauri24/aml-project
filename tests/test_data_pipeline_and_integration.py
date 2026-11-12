"""Unit and integration tests for Member 2: Data pipeline, covariates, splits, and multi-model integration."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.load_data import STATIC
from src.evaluation.backtesting import (
    FUTURE_KEYS, PREDICTION_COLUMNS, ForecastInputs, attach_actuals, construct_inputs,
)
from src.evaluation.integration import (
    MODEL_FAMILIES, compute_comprehensive_metrics, load_all_model_benchmarks,
    verify_consistent_evaluation_inputs,
)
from src.evaluation.splits import development_splits
from src.evaluation.splits_validation import (
    validate_chronological_splits, validate_forecast_inputs_chronology,
    validate_model_internal_splits,
)
from src.evaluation.standardization import standardize_prediction_output
from src.models.deepar_config import DeepARConfig
from src.models.deepar_dataset import (
    DeepARTrainingDataset, StaticCategoryEncoder, extract_dynamic_covariates,
    handle_missing_targets, prepare_deepar_data, validate_deepar_inputs,
)


# =====================================================================
# 1. Missing Observations Handling & Dynamic/Static Covariates
# =====================================================================

def test_handle_missing_targets_policies():
    """Verify missing target imputation policies: zero, forward_fill, and mean."""
    raw = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)

    # Policy 'zero'
    zero_filled = handle_missing_targets(raw, policy="zero")
    np.testing.assert_array_equal(zero_filled, [1.0, 0.0, 3.0, 0.0, 5.0])

    # Policy 'forward_fill'
    ffill = handle_missing_targets(raw, policy="forward_fill")
    np.testing.assert_array_equal(ffill, [1.0, 1.0, 3.0, 3.0, 5.0])

    # Leading NaN with forward_fill falls back to 0.0
    leading_nan = np.array([np.nan, 2.0, np.nan], dtype=np.float32)
    ffill_lead = handle_missing_targets(leading_nan, policy="forward_fill")
    np.testing.assert_array_equal(ffill_lead, [0.0, 2.0, 2.0])

    # Policy 'mean'
    mean_filled = handle_missing_targets(raw, policy="mean")
    np.testing.assert_array_equal(mean_filled, [1.0, 3.0, 3.0, 3.0, 5.0])

    # All NaN falls back to 0.0
    all_nan = np.array([np.nan, np.nan], dtype=np.float32)
    np.testing.assert_array_equal(handle_missing_targets(all_nan, policy="mean"), [0.0, 0.0])

    # Unknown policy raises error
    with pytest.raises(ValueError, match="Unknown missing target policy"):
        handle_missing_targets(raw, policy="invalid_policy")


def test_static_category_encoder_handles_unknown_and_missing():
    """Verify StaticCategoryEncoder maps unknown or missing categories to index 0."""
    train_df = pd.DataFrame({
        "item_id": ["ITEM_1", "ITEM_2"],
        "dept_id": ["DEPT_1", "DEPT_1"],
        "cat_id": ["CAT_1", "CAT_1"],
        "store_id": ["STORE_1", "STORE_2"],
        "state_id": ["CA", "TX"],
    })
    encoder = StaticCategoryEncoder()
    encoder.fit(train_df)

    # Check reserved <unk> index 0
    assert encoder.vocab_sizes_["item_id"] == 3  # 2 unique + <unk>
    assert encoder.mappings_["item_id"]["ITEM_1"] in [1, 2]

    # Test transformation with unknown item and missing value (None / np.nan)
    test_df = pd.DataFrame({
        "item_id": ["ITEM_1", "ITEM_UNKNOWN", None],
        "dept_id": ["DEPT_1", "DEPT_UNKNOWN", "DEPT_1"],
        "cat_id": ["CAT_1", "CAT_1", "CAT_1"],
        "store_id": ["STORE_1", "STORE_2", "STORE_1"],
        "state_id": ["CA", "TX", "CA"],
    })
    encoded = encoder.transform_series(test_df)
    assert encoded["item_id"][0] > 0
    assert encoded["item_id"][1] == 0  # unknown -> 0
    assert encoded["item_id"][2] == 0  # None/NaN -> 0


def test_extract_dynamic_covariates_structure_and_bounds():
    """Verify dynamic calendar covariate extraction produces bounded finite features."""
    df = pd.DataFrame({
        "wday": [1, 2, 7],
        "month": [1, 6, 12],
        "event_name_1": [None, "SuperBowl", np.nan],
        "event_name_2": [np.nan, np.nan, "Easter"],
        "day_num": [1, 100, 1941],
    })
    cov = extract_dynamic_covariates(df)
    assert cov.shape == (3, 8)
    assert np.isfinite(cov).all()
    # wday cyclicality in [-1, 1]
    assert np.all(cov[:, 0] >= -1.0) and np.all(cov[:, 0] <= 1.0)
    assert np.all(cov[:, 1] >= -1.0) and np.all(cov[:, 1] <= 1.0)
    # is_weekend indicators (wday 1 and 2 in M5 are weekend)
    assert cov[0, 4] == 1.0  # wday 1
    assert cov[1, 4] == 1.0  # wday 2
    assert cov[2, 4] == 0.0  # wday 7
    # normalized day progress in [0, 1]
    assert cov[0, 7] == pytest.approx(0.0)
    assert cov[2, 7] == pytest.approx(1940.0 / 1941.0)


# =====================================================================
# 2. DeepAR-Compatible Dataset Preparation & Missing Targets
# =====================================================================

def test_prepare_deepar_data_with_missing_observations(tiny_panel):
    """Verify DeepAR dataset construction handles missing historical sales cleanly."""
    panel, cfg = tiny_panel
    fold = development_splits(cfg)[0]
    inputs = construct_inputs(panel, fold)

    # Introduce synthetic missing observations (NaN) in history
    corrupt_history = inputs.history.copy()
    corrupt_history.loc[corrupt_history["day_num"] == 2, "units_sold"] = np.nan
    corrupt_inputs = ForecastInputs(
        history=corrupt_history,
        static=inputs.static,
        future_known=inputs.future_known,
        origin=inputs.origin,
        horizon=inputs.horizon,
    )

    deepar_cfg = DeepARConfig(
        context_length=4,
        prediction_length=inputs.horizon,
        val_window_length=2,
        windows_per_series_train=2,
        missing_target_policy="zero",
    )

    prepared = prepare_deepar_data(corrupt_inputs, deepar_cfg)
    assert len(prepared.series_ids) == inputs.history["series_id"].nunique()
    assert len(prepared.train_dataset) > 0
    assert len(prepared.val_dataset) > 0

    # Verify train dataset items contain no NaNs
    sample = prepared.train_dataset[0]
    assert torch.is_tensor(sample["target"])
    assert not torch.isnan(sample["target"]).any()
    assert not torch.isnan(sample["covariates"]).any()
    assert sample["scale"] > 0
    assert not torch.isnan(sample["scale"])

    # Verify inference data contains no NaNs
    inf = prepared.inference_data
    assert not torch.isnan(inf["context_targets"]).any()
    assert not torch.isnan(inf["context_covs"]).any()
    assert not torch.isnan(inf["future_covs"]).any()
    assert not torch.isnan(inf["scales"]).any()


# =====================================================================
# 3. Chronological Splits Validation
# =====================================================================

def test_validate_chronological_splits_canonical(tiny_panel):
    """Verify chronological split validator on experiment configuration."""
    panel, cfg = tiny_panel
    report = validate_chronological_splits(cfg, panel=panel)
    assert report["status"] == "VALID"
    assert report["num_folds"] == len(cfg["forecast"]["development_origins"])
    assert report["horizon"] == cfg["forecast"]["horizon"]
    assert report["holdout_origin"] == cfg["forecast"]["final_origin"]


def test_validate_chronological_splits_rejects_leakage():
    """Verify validator rejects non-monotonic origins or holdout overlap."""
    bad_cfg = {
        "forecast": {
            "horizon": 28,
            "final_origin": 1913,
            "development_origins": [1829, 1801],  # Non-monotonic
        },
        "selection": {"cutoff_day": 1801},
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_chronological_splits(bad_cfg)

    # Fold entering holdout
    leak_cfg = {
        "forecast": {
            "horizon": 28,
            "final_origin": 1913,
            "development_origins": [1801, 1890],  # 1890 + 28 = 1918 > 1913
        },
        "selection": {"cutoff_day": 1801},
    }
    with pytest.raises(ValueError, match="reserved final holdout|exceeds final_origin"):
        validate_chronological_splits(leak_cfg)


def test_validate_forecast_inputs_chronology(tiny_panel):
    """Verify ForecastInputs chronological bounds and forbidden column checks."""
    panel, cfg = tiny_panel
    fold = development_splits(cfg)[0]
    inputs = construct_inputs(panel, fold)

    report = validate_forecast_inputs_chronology(inputs)
    assert report["leakage_free"] is True
    assert report["origin"] == fold.origin
    assert report["future_days"] == [fold.origin + 1, fold.origin + fold.horizon]


def test_validate_model_internal_splits():
    """Verify model internal validation window boundary checks."""
    report = validate_model_internal_splits(
        history_length=1801,
        origin=1801,
        val_window_length=28,
        context_length=56,
    )
    assert report["train_end_day"] == 1773
    assert report["val_start_day"] == 1774
    assert report["val_end_day"] == 1801
    assert report["future_leakage"] is False

    with pytest.raises(ValueError, match="insufficient"):
        validate_model_internal_splits(history_length=50, origin=50, val_window_length=28, context_length=56)


# =====================================================================
# 4. Standardize Model Prediction Outputs
# =====================================================================

def test_standardize_prediction_output(tiny_panel):
    """Verify standardizer enforces PREDICTION_COLUMNS schema and quantile ordering."""
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])
    expected_grid = inputs.future_known[FUTURE_KEYS].copy()

    # Raw predictions dataframe
    raw_df = expected_grid.copy()
    raw_df["model"] = "test_forecaster"
    raw_df["y_pred"] = 1.5
    raw_df["q10"] = 0.5
    raw_df["q50"] = 1.5
    raw_df["q90"] = 2.5

    standardized = standardize_prediction_output(raw_df, expected_grid=expected_grid, model_name="test_forecaster")
    assert list(standardized.columns) == PREDICTION_COLUMNS
    assert len(standardized) == len(expected_grid)
    assert (standardized["q10"] <= standardized["q50"]).all()
    assert (standardized["q50"] <= standardized["q90"]).all()


def test_standardize_prediction_output_clips_minor_negative():
    """Verify standardizer clips minor negative float precision underflow."""
    df = pd.DataFrame({
        "model": ["test_m"],
        "series_id": ["S1"],
        "origin": [1801],
        "horizon": [1],
        "target_day_num": [1802],
        "target_date": [pd.Timestamp("2016-01-04")],
        "y_pred": [-1e-7],  # Slight floating point underflow
        "q10": [np.nan],
        "q50": [np.nan],
        "q90": [np.nan],
    })
    clean = standardize_prediction_output(df, model_name="test_m")
    assert clean["y_pred"].iloc[0] == 0.0


def test_standardize_prediction_output_rejects_large_negative():
    """Verify standardizer rejects true negative demand forecasts."""
    df = pd.DataFrame({
        "model": ["test_m"],
        "series_id": ["S1"],
        "origin": [1801],
        "horizon": [1],
        "target_day_num": [1802],
        "target_date": [pd.Timestamp("2016-01-04")],
        "y_pred": [-0.5],
        "q10": [np.nan],
        "q50": [np.nan],
        "q90": [np.nan],
    })
    with pytest.raises(ValueError, match="significantly negative"):
        standardize_prediction_output(df, model_name="test_m")


# =====================================================================
# 5. Multi-Model Integration & Benchmark Consistency
# =====================================================================

def test_verify_consistent_evaluation_inputs_identical_grid(tiny_panel):
    """Verify cross-model consistency checker validates uniform series and folds."""
    panel, cfg = tiny_panel
    folds = development_splits(cfg)
    expected_rows = len(folds) * 240 * cfg["forecast"]["horizon"]

    # Synthesize two mock model prediction frames matching standard contract
    series_ids = [f"SERIES_{i:03d}" for i in range(240)]
    rows = []
    for f in folds:
        for sid in series_ids:
            for h in range(1, cfg["forecast"]["horizon"] + 1):
                rows.append({
                    "series_id": sid,
                    "origin": f.origin,
                    "horizon": h,
                    "target_day_num": f.origin + h,
                    "target_date": pd.Timestamp("2016-01-01") + pd.Timedelta(days=f.origin + h),
                    "y_pred": 1.0,
                    "q10": np.nan,
                    "q50": np.nan,
                    "q90": np.nan,
                })
    base_df = pd.DataFrame(rows)

    m1_df = base_df.copy()
    m1_df["model"] = "model_1"
    m2_df = base_df.copy()
    m2_df["model"] = "model_2"

    report = verify_consistent_evaluation_inputs(
        {"model_1": m1_df, "model_2": m2_df},
        cfg=cfg,
        panel=panel,
    )
    assert report["status"] == "ALL_MODELS_CONSISTENT"
    assert report["expected_rows"] == expected_rows


def test_load_all_model_benchmarks_compiles_four_families():
    """Verify load_all_model_benchmarks loads models from all 4 families and ranks by WAPE."""
    benchmarks = load_all_model_benchmarks(reports_dir="reports")
    assert not benchmarks.empty
    families = set(benchmarks["family"].unique())
    # Must include all four paradigms
    assert "Baseline Heuristic" in families
    assert "State-Space (Kalman)" in families
    assert "Gradient Boosted Trees (LightGBM)" in families
    assert "Deep Learning (DeepAR)" in families

    # Must be ranked ascending by WAPE
    wapes = benchmarks["wape"].tolist()
    assert wapes == sorted(wapes)
    assert benchmarks["point_accuracy_rank"].tolist() == list(range(1, len(benchmarks) + 1))
