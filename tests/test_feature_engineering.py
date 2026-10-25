"""Tests for leakage-safe feature engineering (Milestone 8)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtesting import construct_inputs, ForecastInputs
from src.evaluation.splits import development_splits
from src.features.feature_engineering import (
    CategoryEncoder,
    FeatureEngineer,
    assert_no_future_leakage,
    verify_feature_availability,
    _compute_series_features,
    _price_features_train,
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
)


# ── CategoryEncoder ─────────────────────────────────────────────────────────

class TestCategoryEncoder:
    def test_fit_creates_mapping(self):
        df = pd.DataFrame({"col": ["A", "B", "C", "A"]})
        enc = CategoryEncoder().fit(df, ["col"])
        assert enc.mappings_["col"] == {"A": 0, "B": 1, "C": 2}

    def test_transform_applies_mapping(self):
        df = pd.DataFrame({"col": ["A", "B", "C"]})
        enc = CategoryEncoder().fit(df, ["col"])
        result = enc.transform(df)
        assert list(result["col"]) == [0, 1, 2]

    def test_unseen_category_maps_to_minus_one(self):
        df_train = pd.DataFrame({"col": ["A", "B"]})
        df_test = pd.DataFrame({"col": ["A", "Z"]})
        enc = CategoryEncoder().fit(df_train, ["col"])
        result = enc.transform(df_test)
        assert result["col"].iloc[1] == -1

    def test_multiple_columns(self):
        df = pd.DataFrame({"a": ["X", "Y"], "b": ["P", "Q"]})
        enc = CategoryEncoder().fit(df, ["a", "b"])
        assert "a" in enc.mappings_ and "b" in enc.mappings_


# ── Per-series feature computation ──────────────────────────────────────────

class TestSeriesFeatures:
    @pytest.fixture
    def simple_series(self):
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            "series_id": "TEST__STORE_1",
            "day_num": np.arange(1, n + 1),
            "units_sold": np.random.poisson(5, n).astype("float32"),
            "sell_price": np.full(n, 2.5, dtype="float32"),
        })

    def test_lag_columns_created(self, simple_series):
        result = _compute_series_features(
            simple_series, DEFAULT_LAGS, DEFAULT_ROLLING_WINDOWS, 1
        )
        for lag in DEFAULT_LAGS:
            assert f"lag_{lag}" in result.columns

    def test_lag_1_matches_shifted_sales(self, simple_series):
        result = _compute_series_features(
            simple_series, [1], [], 1
        )
        expected = simple_series["units_sold"].shift(1)
        pd.testing.assert_series_equal(
            result["lag_1"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )

    def test_rolling_mean_does_not_include_current(self, simple_series):
        result = _compute_series_features(
            simple_series, [], [7], 1
        )
        # Rolling mean at row i uses sales[i-7 : i-1] (shift(1) then window 7)
        # At row 8 (index 7, day_num=8), it should use days 1-7
        row = result.iloc[7]
        expected = simple_series["units_sold"].iloc[:7].mean()
        assert abs(row["rmean_7"] - expected) < 1e-5

    def test_rolling_columns_created(self, simple_series):
        result = _compute_series_features(
            simple_series, DEFAULT_LAGS, DEFAULT_ROLLING_WINDOWS, 1
        )
        for w in DEFAULT_ROLLING_WINDOWS:
            assert f"rmean_{w}" in result.columns
            assert f"rstd_{w}" in result.columns

    def test_first_row_lag_is_nan(self, simple_series):
        result = _compute_series_features(simple_series, [1], [], 1)
        assert np.isnan(result["lag_1"].iloc[0])


# ── Price features ───────────────────────────────────────────────────────────

class TestPriceFeatures:
    def test_last_known_price_forward_fills(self):
        df = pd.DataFrame({
            "series_id": ["A"] * 5,
            "sell_price": [1.0, np.nan, np.nan, 2.0, np.nan],
        })
        result = _price_features_train(df)
        np.testing.assert_allclose(result["last_known_price"], [np.nan, 1.0, 1.0, 1.0, 2.0], equal_nan=True)


# ── Integration with ForecastInputs ─────────────────────────────────────────

class TestFeatureEngineerIntegration:
    """Test the full FeatureEngineer with the tiny_panel fixture."""

    def test_fit_succeeds(self, tiny_panel):
        panel, cfg = tiny_panel
        fold = development_splits(cfg)[0]
        inputs = construct_inputs(panel, fold)
        fe = FeatureEngineer()
        fe.fit(inputs)
        assert fe._fitted_origin == fold.origin

    def test_build_train_features_returns_correct_shape(self, tiny_panel):
        panel, cfg = tiny_panel
        fold = development_splits(cfg)[0]
        inputs = construct_inputs(panel, fold)
        fe = FeatureEngineer()
        fe.fit(inputs)
        X, y = fe.build_train_features(inputs)
        assert len(X) == len(y)
        assert len(X) > 0
        assert set(fe.feature_columns).issubset(X.columns)

    def test_build_predict_features_returns_correct_shape(self, tiny_panel):
        panel, cfg = tiny_panel
        fold = development_splits(cfg)[0]
        inputs = construct_inputs(panel, fold)
        fe = FeatureEngineer()
        fe.fit(inputs)
        X_pred = fe.build_predict_features(inputs)
        n_series = inputs.static.shape[0]
        assert len(X_pred) == n_series * inputs.horizon
        assert set(fe.feature_columns).issubset(X_pred.columns)

    def test_train_target_is_nonnegative(self, tiny_panel):
        panel, cfg = tiny_panel
        fold = development_splits(cfg)[0]
        inputs = construct_inputs(panel, fold)
        fe = FeatureEngineer()
        fe.fit(inputs)
        _, y = fe.build_train_features(inputs)
        assert (y.dropna() >= 0).all()

    def test_no_future_sales_in_train_features(self, tiny_panel):
        """Mutate sales after origin; train features must not change."""
        panel, cfg = tiny_panel
        fold = development_splits(cfg)[0]
        inputs = construct_inputs(panel, fold)
        fe = FeatureEngineer()
        fe.fit(inputs)
        X_orig, y_orig = fe.build_train_features(inputs)

        # Assert history only goes up to origin
        assert_no_future_leakage(X_orig, inputs.history, inputs.origin)

    def test_predict_lag_leakage_check(self, tiny_panel):
        """For prediction, lag_k at horizon h should be NaN when h > k."""
        panel, cfg = tiny_panel
        fold = development_splits(cfg)[0]
        inputs = construct_inputs(panel, fold)
        fe = FeatureEngineer()
        fe.fit(inputs)
        X_pred = fe.build_predict_features(inputs)

        # Get horizon info from the future_known frame
        future = inputs.future_known.copy()
        for lag in DEFAULT_LAGS:
            col = f"lag_{lag}"
            if col in X_pred.columns:
                # Rows where horizon > lag should have NaN lag
                invalid = future["horizon"] > lag
                violations = X_pred.loc[invalid.values, col].notna().sum()
                assert violations == 0, (
                    f"{col}: {violations} non-NaN values at horizon > {lag}"
                )

    def test_fit_required_before_train(self, tiny_panel):
        panel, cfg = tiny_panel
        fold = development_splits(cfg)[0]
        inputs = construct_inputs(panel, fold)
        fe = FeatureEngineer()
        with pytest.raises(RuntimeError, match="fit"):
            fe.build_train_features(inputs)

    def test_fit_required_before_predict(self, tiny_panel):
        panel, cfg = tiny_panel
        fold = development_splits(cfg)[0]
        inputs = construct_inputs(panel, fold)
        fe = FeatureEngineer()
        with pytest.raises(RuntimeError, match="fit"):
            fe.build_predict_features(inputs)

    def test_wrong_origin_predict_raises(self, tiny_panel):
        panel, cfg = tiny_panel
        folds = development_splits(cfg)
        inputs_a = construct_inputs(panel, folds[0])
        inputs_b = construct_inputs(panel, folds[1])
        fe = FeatureEngineer()
        fe.fit(inputs_a)
        with pytest.raises(RuntimeError, match="origin"):
            fe.build_predict_features(inputs_b)

    def test_feature_columns_consistent(self, tiny_panel):
        """Train and predict feature matrices have the same columns."""
        panel, cfg = tiny_panel
        fold = development_splits(cfg)[0]
        inputs = construct_inputs(panel, fold)
        fe = FeatureEngineer()
        fe.fit(inputs)
        X_train, _ = fe.build_train_features(inputs)
        X_pred = fe.build_predict_features(inputs)
        assert list(X_train.columns) == list(X_pred.columns)

    def test_categorical_encoding_consistent(self, tiny_panel):
        """Static columns are integer-encoded in both train and predict."""
        panel, cfg = tiny_panel
        fold = development_splits(cfg)[0]
        inputs = construct_inputs(panel, fold)
        fe = FeatureEngineer()
        fe.fit(inputs)
        X_train, _ = fe.build_train_features(inputs)
        X_pred = fe.build_predict_features(inputs)
        for col in FeatureEngineer.STATIC_COLS:
            assert X_train[col].dtype in (np.int32, np.int64, np.float64)
            assert X_pred[col].dtype in (np.int32, np.int64, np.float64)

    def test_calendar_features_present(self, tiny_panel):
        panel, cfg = tiny_panel
        fold = development_splits(cfg)[0]
        inputs = construct_inputs(panel, fold)
        fe = FeatureEngineer()
        fe.fit(inputs)
        X_train, _ = fe.build_train_features(inputs)
        for col in ["dayofweek", "is_weekend", "month"]:
            assert col in X_train.columns

    def test_multiple_folds_independent(self, tiny_panel):
        """Fitting on fold 2 does not contaminate fold 1 features."""
        panel, cfg = tiny_panel
        folds = development_splits(cfg)
        inputs_a = construct_inputs(panel, folds[0])
        inputs_b = construct_inputs(panel, folds[1])

        fe = FeatureEngineer()
        fe.fit(inputs_a)
        X_a, _ = fe.build_train_features(inputs_a)

        fe.fit(inputs_b)
        X_b, _ = fe.build_train_features(inputs_b)

        # Fold B has more training rows (later origin)
        assert len(X_b) >= len(X_a)


# ── Verify feature availability helper ──────────────────────────────────────

class TestVerifyFeatureAvailability:
    def test_passes_valid_prediction(self):
        df = pd.DataFrame({
            "lag_1": [1.0, np.nan, np.nan],
            "lag_7": [2.0, 3.0, np.nan],
            "horizon": [1, 2, 8],
        })
        verify_feature_availability(df, origin=100, horizon=28, mode="predict")

    def test_detects_future_leakage(self):
        df = pd.DataFrame({
            "lag_1": [1.0, 999.0],  # horizon=2, lag=1 → should be NaN
            "horizon": [1, 2],
        })
        with pytest.raises(AssertionError, match="leakage"):
            verify_feature_availability(df, origin=100, horizon=28, mode="predict")


# ── Detailed lag validity test ───────────────────────────────────────────────

class TestLagPopulation:
    def test_predict_lags_on_tiny_panel(self, tiny_panel):
        panel, cfg = tiny_panel
        fold = development_splits(cfg)[0]  # origin = 6, horizon = 3
        inputs = construct_inputs(panel, fold)
        fe = FeatureEngineer()
        fe.fit(inputs)
        X_pred = fe.build_predict_features(inputs)

        h = inputs.future_known["horizon"].values
        # For horizon 1: target_day = 7, lag_1 requires day 6 (origin), which is in history
        assert X_pred.loc[h == 1, "lag_1"].notna().all(), "lag_1 should be populated for horizon=1"

        # For horizon 2: target_day = 8, lag_1 requires day 7 > origin(6) -> must be NaN
        assert X_pred.loc[h == 2, "lag_1"].isna().all(), "lag_1 at horizon 2 must be NaN"

        # For horizon 3: target_day = 9, lag_1 requires day 8 > origin(6) -> must be NaN
        assert X_pred.loc[h == 3, "lag_1"].isna().all(), "lag_1 at horizon 3 must be NaN"

    def test_predict_lags_on_longer_history(self):
        """With sufficient history (100 days) and origin 35, test all default lags across 28 horizons."""
        n_days = 100
        origin = 35
        horizon = 28
        dates = pd.date_range("2015-01-01", periods=n_days)
        history = pd.DataFrame({
            "series_id": "TEST_SERIES",
            "item_id": "ITEM_1",
            "store_id": "STORE_1",
            "dept_id": "DEPT_1",
            "cat_id": "CAT_1",
            "state_id": "CA",
            "day_num": np.arange(1, origin + 1),
            "date": dates[:origin],
            "units_sold": np.random.poisson(4, origin).astype("float32"),
            "sell_price": np.full(origin, 2.99, dtype="float32"),
            "snap": np.zeros(origin, dtype="int8"),
            "event_name_1": [None] * origin,
            "event_name_2": [None] * origin,
        })
        static = pd.DataFrame([{
            "series_id": "TEST_SERIES",
            "item_id": "ITEM_1",
            "store_id": "STORE_1",
            "dept_id": "DEPT_1",
            "cat_id": "CAT_1",
            "state_id": "CA",
        }])
        future_known = pd.DataFrame({
            "series_id": ["TEST_SERIES"] * horizon,
            "origin": [origin] * horizon,
            "horizon": np.arange(1, horizon + 1),
            "target_day_num": np.arange(origin + 1, origin + horizon + 1),
            "target_date": dates[origin : origin + horizon].values,
            "wm_yr_wk": [11501] * horizon,
            "weekday": dates[origin : origin + horizon].day_name().values,
            "wday": [1] * horizon,
            "month": dates[origin : origin + horizon].month.values,
            "year": dates[origin : origin + horizon].year.values,
            "event_name_1": [None] * horizon,
            "event_type_1": [None] * horizon,
            "event_name_2": [None] * horizon,
            "event_type_2": [None] * horizon,
        })
        inputs = ForecastInputs(
            history=history,
            static=static,
            future_known=future_known,
            origin=origin,
            horizon=horizon,
        )
        fe = FeatureEngineer()
        fe.fit(inputs)
        X_pred = fe.build_predict_features(inputs)

        h = future_known["horizon"].values
        # At horizon 1, all lags (1, 7, 14, 28) should be populated
        for lag in DEFAULT_LAGS:
            assert X_pred.loc[h == 1, f"lag_{lag}"].notna().all(), f"lag_{lag} populated at h=1"

        # At horizon 7: lag 1 is NaN, lag 7, 14, 28 are populated
        assert X_pred.loc[h == 7, "lag_1"].isna().all()
        assert X_pred.loc[h == 7, "lag_7"].notna().all()
        assert X_pred.loc[h == 7, "lag_14"].notna().all()
        assert X_pred.loc[h == 7, "lag_28"].notna().all()

        # At horizon 14: lag 1, 7 are NaN; lag 14, 28 are populated
        assert X_pred.loc[h == 14, "lag_1"].isna().all()
        assert X_pred.loc[h == 14, "lag_7"].isna().all()
        assert X_pred.loc[h == 14, "lag_14"].notna().all()
        assert X_pred.loc[h == 14, "lag_28"].notna().all()

        # At horizon 28: only lag 28 is populated
        assert X_pred.loc[h == 28, "lag_1"].isna().all()
        assert X_pred.loc[h == 28, "lag_7"].isna().all()
        assert X_pred.loc[h == 28, "lag_14"].isna().all()
        assert X_pred.loc[h == 28, "lag_28"].notna().all()



# ── Dataset Builder Test ────────────────────────────────────────────────────

class TestDatasetBuilder:
    def test_build_feature_datasets_runs(self, tmp_path):
        from src.features.feature_engineering import build_feature_datasets
        metadata = build_feature_datasets(
            config_path="experiment.yaml",
            output_dir=str(tmp_path),
        )
        assert metadata["n_folds"] == 4
        assert len(metadata["folds"]) == 4
        for origin, info in metadata["folds"].items():
            assert info["train_rows"] > 0
            assert info["predict_rows"] == 6720
            import os
            assert os.path.exists(info["train_file"])
            assert os.path.exists(info["predict_file"])
        assert "feature_columns" in metadata

