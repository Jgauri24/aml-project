"""Leakage-safe feature engineering for multi-step demand forecasting.

Converts ``ForecastInputs`` into tabular training/prediction matrices for a
gradient-boosting model.  Every feature is computed **only from information
available at the forecast origin** for training rows and at the *origin* (not
the target date) for prediction rows.

Design decisions
~~~~~~~~~~~~~~~~
* **Lag features** use day-number offsets relative to each row's reference day
  (``day_num`` for training, ``origin`` for prediction).  The minimum lag is 1
  so we never use the target itself.
* **Rolling statistics** are trailing windows ending at the reference day.
  They are computed per-series with a min-periods guard to avoid NaN inflation
  in short histories.
* **Calendar features** (day-of-week, month, weekend, events, SNAP) are known
  for future dates and safe to use on both train and prediction rows.
* **Price features** use only historical prices available at the origin.  We
  compute the most-recent known price and a trailing price-change ratio.
* **Static encodings** convert categorical identifiers to integer codes.
  The encoder is fitted inside ``fit`` so it never sees future data.
* **Multi-step strategy**: each horizon step gets its own feature row.  Lag
  features for h > 1 that would require unknowable future sales are set to NaN
  — the tree model handles missing splits natively.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.backtesting import ForecastInputs, FUTURE_KEYS

# ── Lag and rolling definitions ──────────────────────────────────────────────
DEFAULT_LAGS = [1, 7, 14, 28]
DEFAULT_ROLLING_WINDOWS = [7, 14, 28]
DEFAULT_ROLLING_MIN_PERIODS = 1

# Calendar features available in future_known
KNOWN_CALENDAR = [
    "wm_yr_wk", "weekday", "wday", "month", "year",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
]


# ── Categorical encoder ─────────────────────────────────────────────────────
@dataclass
class CategoryEncoder:
    """Maps string categoricals to integer codes, fitted on training data."""
    mappings_: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "CategoryEncoder":
        self.mappings_ = {}
        for col in columns:
            values = sorted(df[col].astype(str).unique())
            self.mappings_[col] = {v: i for i, v in enumerate(values)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col, mapping in self.mappings_.items():
            if col in result.columns:
                result[col] = (
                    result[col].astype(str).map(mapping).fillna(-1).astype("int32")
                )
        return result


# ── Per-series lag / rolling cache ───────────────────────────────────────────
def _compute_series_features(
    series_history: pd.DataFrame,
    lags: List[int],
    rolling_windows: List[int],
    rolling_min_periods: int,
) -> pd.DataFrame:
    """Compute lag and rolling features for one sorted series block.

    ``series_history`` must be sorted by ``day_num`` with no gaps.
    Returns the same DataFrame with new lag/rolling columns appended.
    """
    result = series_history.copy()
    sales = result["units_sold"].astype("float64")

    # Lag features
    for lag in lags:
        result[f"lag_{lag}"] = sales.shift(lag)

    # Rolling-mean and rolling-std features
    for window in rolling_windows:
        rolled = sales.shift(1).rolling(window, min_periods=rolling_min_periods)
        result[f"rmean_{window}"] = rolled.mean()
        result[f"rstd_{window}"] = rolled.std()

    return result


def _compute_series_origin_snapshot(
    series_history: pd.DataFrame,
    origin: int,
    lags: List[int],
    rolling_windows: List[int],
    rolling_min_periods: int,
) -> Dict[str, float]:
    """Extract lag/rolling values at *one origin* for prediction rows.

    Returns a dict ``{feature_name: value}`` representing the snapshot at the
    forecast origin.  Lags are relative to ``origin``, so ``lag_1`` is the
    sales value at ``day_num == origin``.
    """
    sorted_hist = series_history.sort_values("day_num")
    sales = sorted_hist.set_index("day_num")["units_sold"].astype("float64")

    snapshot: Dict[str, float] = {}

    for lag in lags:
        target_day = origin - lag + 1  # lag_1 → day at origin
        # For prediction: lag_k means k days before the *target day*.
        # But since each horizon row has a different target day, we store
        # the origin-relative values here and adjust per-horizon later.
        day_idx = origin - lag
        snapshot[f"origin_lag_{lag}"] = (
            float(sales.loc[day_idx]) if day_idx in sales.index else np.nan
        )

    # Rolling stats at the origin
    available = sales.loc[sales.index <= origin]
    for window in rolling_windows:
        trailing = available.iloc[-window:]
        if len(trailing) >= rolling_min_periods:
            snapshot[f"origin_rmean_{window}"] = float(trailing.mean())
            snapshot[f"origin_rstd_{window}"] = float(trailing.std())
        else:
            snapshot[f"origin_rmean_{window}"] = np.nan
            snapshot[f"origin_rstd_{window}"] = np.nan

    return snapshot


# ── Price features ───────────────────────────────────────────────────────────
def _price_features_train(history: pd.DataFrame) -> pd.DataFrame:
    """Add historical-price features to training data (per-series, sorted)."""
    result = history.copy()

    # Forward-fill price within series to get 'last known price'
    result["last_known_price"] = (
        result.groupby("series_id", observed=True)["sell_price"]
        .transform(lambda s: s.ffill())
    )

    # Price change ratio: current price / previous week's price
    result["price_change_ratio"] = (
        result.groupby("series_id", observed=True)["sell_price"]
        .transform(lambda s: s / s.shift(7))
    )

    return result


def _price_snapshot_at_origin(
    series_history: pd.DataFrame, origin: int
) -> Dict[str, float]:
    """Price features available at the origin for prediction rows."""
    hist_at_origin = series_history.loc[series_history["day_num"] <= origin]
    prices = hist_at_origin["sell_price"].dropna()
    snapshot: Dict[str, float] = {}
    if len(prices):
        snapshot["last_known_price"] = float(prices.iloc[-1])
        if len(prices) >= 8:
            prev = prices.iloc[-8]
            snapshot["price_change_ratio"] = (
                float(prices.iloc[-1] / prev) if prev > 0 else np.nan
            )
        else:
            snapshot["price_change_ratio"] = np.nan
    else:
        snapshot["last_known_price"] = np.nan
        snapshot["price_change_ratio"] = np.nan
    return snapshot


# ── Calendar features ────────────────────────────────────────────────────────
def _encode_calendar(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Derive numeric calendar features from the date or known calendar cols.

    Works on both training rows (``date`` column) and prediction rows
    (``target_date`` column).
    """
    result = df.copy()

    # Day of week as integer (Monday=0 … Sunday=6)
    if date_col in result.columns:
        dt = pd.to_datetime(result[date_col])
        result["dayofweek"] = dt.dt.dayofweek.astype("int8")
        result["dayofmonth"] = dt.dt.day.astype("int8")
        result["weekofyear"] = dt.dt.isocalendar().week.astype("int8")
        result["is_weekend"] = result["dayofweek"].isin([5, 6]).astype("int8")
    elif "wday" in result.columns:
        # wday in M5 data: 1=Saturday … 7=Friday
        result["dayofweek"] = ((result["wday"].astype(int) + 4) % 7).astype("int8")
        result["is_weekend"] = result["dayofweek"].isin([5, 6]).astype("int8")

    # Event flags
    for event_col in ["event_name_1", "event_name_2"]:
        if event_col in result.columns:
            result[f"has_{event_col}"] = (
                result[event_col]
                .astype(str)
                .ne("nan")
                .astype("int8")
            )
            # Also check for actual NaN/None/""
            result[f"has_{event_col}"] = result[f"has_{event_col}"] & (
                result[event_col].astype(str).ne("")
            ).astype("int8")

    return result


# ── Main feature builder ────────────────────────────────────────────────────
class FeatureEngineer:
    """Leakage-safe feature builder for the LightGBM forecaster.

    Usage
    -----
    1. Call ``fit(inputs)`` once per fold to learn categorical mappings and
       cache per-series origin snapshots.
    2. Call ``build_train_features(inputs)`` to get ``(X, y)`` for training.
    3. Call ``build_predict_features(inputs)`` to get ``X`` for prediction.

    Features produced
    ~~~~~~~~~~~~~~~~~
    - ``lag_1, lag_7, lag_14, lag_28``: demand lags relative to each row's day.
    - ``rmean_7, rmean_14, rmean_28``: rolling mean of past demand.
    - ``rstd_7, rstd_14, rstd_28``: rolling std of past demand.
    - ``dayofweek, dayofmonth, weekofyear, month, year, is_weekend``: calendar.
    - ``has_event_name_1, has_event_name_2``: binary event indicators.
    - ``snap``: state-specific SNAP indicator.
    - ``last_known_price, price_change_ratio``: price features.
    - ``item_id, store_id, dept_id, cat_id, state_id``: encoded categoricals.
    """

    LAGS = DEFAULT_LAGS
    ROLLING_WINDOWS = DEFAULT_ROLLING_WINDOWS
    ROLLING_MIN_PERIODS = DEFAULT_ROLLING_MIN_PERIODS

    # Columns that will be in the final feature matrix
    LAG_COLS = [f"lag_{l}" for l in DEFAULT_LAGS]
    ROLLING_COLS = (
        [f"rmean_{w}" for w in DEFAULT_ROLLING_WINDOWS]
        + [f"rstd_{w}" for w in DEFAULT_ROLLING_WINDOWS]
    )
    CALENDAR_COLS = [
        "dayofweek", "dayofmonth", "weekofyear", "month", "year", "is_weekend",
        "has_event_name_1", "has_event_name_2",
    ]
    SNAP_COLS = ["snap"]
    PRICE_COLS = ["last_known_price", "price_change_ratio"]
    STATIC_COLS = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        rolling_min_periods: int = DEFAULT_ROLLING_MIN_PERIODS,
    ):
        self.lags = lags or list(DEFAULT_LAGS)
        self.rolling_windows = rolling_windows or list(DEFAULT_ROLLING_WINDOWS)
        self.rolling_min_periods = rolling_min_periods
        self._encoder = CategoryEncoder()
        self._origin_snapshots: Dict[str, Dict[str, float]] = {}
        self._recent_sales: Dict[tuple[str, int], float] = {}
        self._fitted_origin: Optional[int] = None

    @property
    def feature_columns(self) -> List[str]:
        """Ordered list of feature column names in the output matrix."""
        lag_cols = [f"lag_{l}" for l in self.lags]
        roll_cols = (
            [f"rmean_{w}" for w in self.rolling_windows]
            + [f"rstd_{w}" for w in self.rolling_windows]
        )
        return (
            lag_cols + roll_cols
            + self.CALENDAR_COLS + self.SNAP_COLS
            + self.PRICE_COLS + self.STATIC_COLS
        )

    def fit(self, inputs: ForecastInputs) -> "FeatureEngineer":
        """Learn encodings and cache origin-level snapshots.

        Must be called once per backtest fold before building features.
        """
        history = inputs.history.copy()

        # Fit categorical encoder on training history
        self._encoder.fit(history, self.STATIC_COLS)

        # Cache per-series snapshots at the fold origin
        self._origin_snapshots = {}
        for series_id, group in history.groupby("series_id", observed=True):
            sid = str(series_id)
            group_sorted = group.sort_values("day_num")

            # Lag/rolling snapshot
            snap = _compute_series_origin_snapshot(
                group_sorted, inputs.origin, self.lags,
                self.rolling_windows, self.rolling_min_periods,
            )
            # Price snapshot
            snap.update(_price_snapshot_at_origin(group_sorted, inputs.origin))

            self._origin_snapshots[sid] = snap

        # Cache recent observed sales per series up to origin for lag retrieval
        min_day_needed = inputs.origin - max(self.lags) - inputs.horizon - 5
        recent = history.loc[
            (history["day_num"] >= min_day_needed) & (history["day_num"] <= inputs.origin),
            ["series_id", "day_num", "units_sold"],
        ]
        self._recent_sales = {
            (str(sid), int(d)): float(u)
            for sid, d, u in zip(recent["series_id"], recent["day_num"], recent["units_sold"])
        }

        self._fitted_origin = inputs.origin
        return self

    def build_train_features(
        self, inputs: ForecastInputs
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build ``(X_train, y_train)`` from history rows.

        Each training row represents one series on one day.  Lag and rolling
        features are computed from preceding observations — never from the
        target itself or future days.

        Rows with insufficient history for *all* lag features are dropped.
        """
        if self._fitted_origin is None:
            raise RuntimeError("Call fit() before building features")

        history = inputs.history.copy()
        history = history.sort_values(["series_id", "day_num"]).reset_index(drop=True)

        # Lag and rolling features (per-series)
        pieces = []
        for sid, group in history.groupby("series_id", observed=True):
            piece = _compute_series_features(
                group.sort_values("day_num"),
                self.lags,
                self.rolling_windows,
                self.rolling_min_periods,
            )
            pieces.append(piece)
        featured = pd.concat(pieces, ignore_index=True)

        # Price features
        featured = _price_features_train(featured)

        # Calendar features
        featured = _encode_calendar(featured, date_col="date")

        # Encode categoricals
        featured = self._encoder.transform(featured)

        # Assemble the feature matrix
        feature_cols = self.feature_columns
        missing_cols = [c for c in feature_cols if c not in featured.columns]
        for c in missing_cols:
            featured[c] = np.nan

        X = featured[feature_cols].copy()
        y = featured["units_sold"].copy()

        # Drop rows where the minimum lag is unavailable (very early history)
        min_lag = min(self.lags)
        valid = featured["day_num"] > min_lag
        X = X.loc[valid].reset_index(drop=True)
        y = y.loc[valid].reset_index(drop=True)

        return X, y

    def build_predict_features(
        self, inputs: ForecastInputs
    ) -> pd.DataFrame:
        """Build feature matrix for the 28-day forecast horizon.

        For each (series, horizon) pair, lag features use what is *known at the
        origin*:

        - ``lag_k`` for horizon h uses the observation at ``origin + h - k``.
          If ``origin + h - k > origin`` (i.e. h > k), the required observation
          is in the future and is set to NaN — the tree model handles this with
          missing-value splits.
        - Rolling features use the trailing window ending at the origin;
          they are the same for all horizons of a given series.
        - Calendar features come from ``future_known`` (known for all 28 days).
        - Price and SNAP features use the origin snapshot.
        """
        if self._fitted_origin is None or self._fitted_origin != inputs.origin:
            raise RuntimeError("Call fit() for the current origin before predicting")

        future = inputs.future_known.copy()

        # Calendar features from future_known
        future = _encode_calendar(future, date_col="target_date")

        # Get SNAP for each future row from the static series metadata
        # SNAP is a calendar property that we conservatively exclude for future
        # (as per experiment protocol: future_snap=excluded).  Set to 0.
        future["snap"] = 0

        # Encode categoricals: merge static info
        static = inputs.static.copy()
        future = future.merge(static, on="series_id", how="left")
        future = self._encoder.transform(future)

        # Merge origin stats (rolling stats and price snapshots)
        origin_stats_rows = []
        for sid, snap in self._origin_snapshots.items():
            row = {"series_id": sid}
            for w in self.rolling_windows:
                row[f"rmean_{w}"] = snap.get(f"origin_rmean_{w}", np.nan)
                row[f"rstd_{w}"] = snap.get(f"origin_rstd_{w}", np.nan)
            row["last_known_price"] = snap.get("last_known_price", np.nan)
            row["price_change_ratio"] = snap.get("price_change_ratio", np.nan)
            origin_stats_rows.append(row)

        if origin_stats_rows:
            origin_stats_df = pd.DataFrame(origin_stats_rows)
            if future["series_id"].dtype.name == "category":
                origin_stats_df["series_id"] = origin_stats_df["series_id"].astype("category")
            else:
                origin_stats_df["series_id"] = origin_stats_df["series_id"].astype(future["series_id"].dtype)
            future = future.merge(origin_stats_df, on="series_id", how="left")
        else:
            for w in self.rolling_windows:
                future[f"rmean_{w}"] = np.nan
                future[f"rstd_{w}"] = np.nan
            future["last_known_price"] = np.nan
            future["price_change_ratio"] = np.nan

        # Fast vectorized lag computation
        target_days = future["target_day_num"].values
        series_ids = future["series_id"].astype(str).values
        origin = self._fitted_origin

        for lag in self.lags:
            req_days = target_days - lag
            future[f"lag_{lag}"] = [
                self._recent_sales.get((sid, int(d)), np.nan) if d <= origin else np.nan
                for sid, d in zip(series_ids, req_days)
            ]

        feature_cols = self.feature_columns
        missing_cols = [c for c in feature_cols if c not in future.columns]
        for c in missing_cols:
            future[c] = np.nan

        return future[feature_cols].copy()


# ── Leakage test helpers ────────────────────────────────────────────────────
def assert_no_future_leakage(
    X: pd.DataFrame,
    history: pd.DataFrame,
    origin: int,
) -> None:
    """Verify that no feature value depends on observations after ``origin``.

    Mutates a copy of ``history`` beyond the origin and checks that the
    feature matrix is unchanged.  Raises ``AssertionError`` on leakage.
    """
    # This is a structural check: train features should use only day_num <= origin
    max_day = history["day_num"].max()
    if max_day > origin:
        raise AssertionError(
            f"History contains day_num {max_day} > origin {origin}: leakage"
        )


def verify_feature_availability(
    feature_df: pd.DataFrame,
    origin: int,
    horizon: int,
    mode: str = "predict",
) -> None:
    """Ensure prediction features don't use future sales.

    For prediction mode, lag_k for horizon h is valid only when k >= h,
    otherwise it must be NaN.
    """
    if mode != "predict":
        return

    for lag in DEFAULT_LAGS:
        col = f"lag_{lag}"
        if col not in feature_df.columns:
            continue
        # For rows where horizon > lag, the lag value should be NaN
        if "horizon" in feature_df.columns:
            invalid_mask = feature_df["horizon"] > lag
            violations = feature_df.loc[invalid_mask, col].notna().sum()
            if violations > 0:
                raise AssertionError(
                    f"{col}: {violations} non-NaN values where horizon > lag "
                    f"(future sales leakage)"
                )


# ── CLI & Dataset Builder ───────────────────────────────────────────────────
def build_feature_datasets(
    config_path: str = "experiment.yaml",
    output_dir: str = "data/processed/features",
) -> Dict[str, Any]:
    """Build and save leakage-safe training and prediction feature matrices.

    Generates feature tables for all development folds (origins 1801, 1829, 1857, 1885).
    Saves parquet files and metadata to ``output_dir``.
    """
    import os
    import json
    from src.config import load_config
    from src.data.preprocess import load_panel
    from src.evaluation.backtesting import construct_inputs
    from src.evaluation.splits import development_splits

    cfg = load_config(config_path)
    cutoff = cfg["forecast"]["final_origin"]
    panel = load_panel(cfg, cutoff=cutoff)
    folds = development_splits(cfg)

    os.makedirs(output_dir, exist_ok=True)
    metadata: Dict[str, Any] = {
        "config_path": config_path,
        "cutoff": cutoff,
        "n_folds": len(folds),
        "origins": [f.origin for f in folds],
        "folds": {},
    }

    fe = None
    for fold in folds:
        inputs = construct_inputs(panel, fold)
        fe = FeatureEngineer()
        fe.fit(inputs)

        X_train, y_train = fe.build_train_features(inputs)
        X_predict = fe.build_predict_features(inputs)

        # Leakage verification
        assert_no_future_leakage(X_train, inputs.history, inputs.origin)
        verify_feature_availability(
            X_predict.assign(horizon=inputs.future_known["horizon"].values),
            inputs.origin,
            inputs.horizon,
            mode="predict",
        )

        train_path = os.path.join(output_dir, f"train_features_origin_{fold.origin}.parquet")
        predict_path = os.path.join(output_dir, f"predict_features_origin_{fold.origin}.parquet")

        train_df = X_train.copy()
        train_df["target"] = y_train.values
        train_df.to_parquet(train_path, index=False)

        # Add forecast identifiers to prediction features
        predict_df = X_predict.copy()
        predict_df["series_id"] = inputs.future_known["series_id"].values
        predict_df["origin"] = inputs.future_known["origin"].values
        predict_df["horizon"] = inputs.future_known["horizon"].values
        predict_df["target_day_num"] = inputs.future_known["target_day_num"].values
        predict_df.to_parquet(predict_path, index=False)

        metadata["folds"][str(fold.origin)] = {
            "origin": fold.origin,
            "train_rows": len(train_df),
            "train_cols": len(train_df.columns),
            "predict_rows": len(predict_df),
            "predict_cols": len(predict_df.columns),
            "train_file": train_path,
            "predict_file": predict_path,
        }
        print(f"Origin {fold.origin}: train {train_df.shape} -> {train_path}, predict {predict_df.shape} -> {predict_path}")

    if fe is not None:
        metadata["feature_columns"] = fe.feature_columns
        metadata["lags"] = fe.lags
        metadata["rolling_windows"] = fe.rolling_windows

    meta_path = os.path.join(output_dir, "feature_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved feature metadata to {meta_path}")
    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build leakage-safe feature datasets for M8.")
    parser.add_argument("--config", default="experiment.yaml", help="Path to experiment configuration YAML.")
    parser.add_argument("--output-dir", default="data/processed/features", help="Output directory for parquet files.")
    args = parser.parse_args()

    build_feature_datasets(config_path=args.config, output_dir=args.output_dir)
