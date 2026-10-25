"""Origin-safe one-step and direct multi-horizon features for daily sales.

Future calendar fields are known ahead. Sales summaries, prices and historical
SNAP are frozen at each forecast origin. Future actual prices/SNAP are excluded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.backtesting import ForecastInputs, FUTURE_KEYS, KNOWN_CALENDAR

DEFAULT_LAGS = [1, 7, 14, 28]
DEFAULT_ROLLING_WINDOWS = [7, 14, 28]
DEFAULT_ROLLING_MIN_PERIODS = 1
FEATURE_VERSION = 2
TRAINING_STRATEGY = "direct_nonoverlapping_origin_blocks"


@dataclass
class CategoryEncoder:
    """Fold-local categorical mappings; unknown values map to -1."""
    mappings_: dict[str, dict[str, int]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, columns: list[str]) -> CategoryEncoder:
        self.mappings_ = {col: {value: i for i, value in enumerate(sorted(df[col].astype(str).unique()))}
                          for col in columns}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col, mapping in self.mappings_.items():
            if col in result:
                result[col] = result[col].astype(str).map(mapping).fillna(-1).astype("int32")
        return result


def _compute_series_features(series_history, lags, rolling_windows, rolling_min_periods):
    """One-step sales features at day t use observations through t-1."""
    result = series_history.copy()
    sales = result.units_sold.astype("float64")
    for lag in lags:
        result[f"lag_{lag}"] = sales.shift(lag)
    for window in rolling_windows:
        rolled = sales.shift(1).rolling(window, min_periods=rolling_min_periods)
        result[f"rmean_{window}"] = rolled.mean()
        result[f"rstd_{window}"] = rolled.std()
    return result


def _compute_series_origin_snapshot(series_history, origin, lags, rolling_windows, rolling_min_periods):
    """Observed lag and rolling values ending at the origin, never after it."""
    sales = series_history.sort_values("day_num").set_index("day_num").units_sold.astype(float)
    available = sales.loc[sales.index <= origin]
    snapshot = {f"origin_lag_{lag}": float(available.get(origin - lag + 1, np.nan)) for lag in lags}
    for window in rolling_windows:
        trailing = available.iloc[-window:]
        enough = trailing.notna().sum() >= rolling_min_periods
        snapshot[f"origin_rmean_{window}"] = float(trailing.mean()) if enough else np.nan
        snapshot[f"origin_rstd_{window}"] = float(trailing.std()) if enough else np.nan
    return snapshot


def _price_features_train(history: pd.DataFrame) -> pd.DataFrame:
    """At label day t, only prices known through t-1 are available."""
    result = history.copy()
    result["last_known_price"] = result.groupby("series_id", observed=True).sell_price.transform(
        lambda s: s.ffill().shift(1))
    result["price_change_ratio"] = result.groupby("series_id", observed=True).last_known_price.transform(
        lambda s: s / s.shift(7).where(s.shift(7).gt(0)))
    return result


def _price_snapshot_at_origin(series_history: pd.DataFrame, origin: int) -> dict[str, float]:
    """As-of prices at origin and origin-7, including calendar gaps in the lookup."""
    hist = series_history.loc[series_history.day_num.le(origin)].sort_values("day_num")
    prices = hist.set_index("day_num").sell_price.reindex(range(1, origin + 1)).ffill()
    latest, previous = float(prices.loc[origin]), float(prices.get(origin - 7, np.nan))
    return {"last_known_price": latest,
            "price_change_ratio": latest / previous if previous > 0 else np.nan}


def _encode_calendar(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Known calendar values; nulls and blank event fields mean no event."""
    result = df.copy()
    if date_col in result:
        dates = pd.to_datetime(result[date_col])
        result["dayofweek"] = dates.dt.dayofweek.astype("int8")
        result["dayofmonth"] = dates.dt.day.astype("int8")
        result["weekofyear"] = dates.dt.isocalendar().week.astype("int8")
        result["is_weekend"] = result.dayofweek.isin([5, 6]).astype("int8")
    elif "wday" in result:
        result["dayofweek"] = ((result.wday.astype(int) + 4) % 7).astype("int8")
        result["is_weekend"] = result.dayofweek.isin([5, 6]).astype("int8")
    for column in ["event_name_1", "event_name_2"]:
        if column in result:
            values = result[column].astype("string").str.strip().str.lower().fillna("")
            result[f"has_{column}"] = (result[column].notna()
                & ~values.isin(["", "nan", "none", "<na>", "nat"])).astype("int8")
    return result


class FeatureEngineer:
    """Fit categorical mappings and frozen per-series inference state per fold.

    ``build_train_features`` retains the one-step API. M9 and dataset export use
    ``build_direct_train_features`` so train and inference horizons have the same
    origin-based feature contract.
    """
    LAGS = DEFAULT_LAGS
    ROLLING_WINDOWS = DEFAULT_ROLLING_WINDOWS
    ROLLING_MIN_PERIODS = DEFAULT_ROLLING_MIN_PERIODS
    LAG_COLS = [f"lag_{lag}" for lag in DEFAULT_LAGS]
    ROLLING_COLS = [f"rmean_{w}" for w in DEFAULT_ROLLING_WINDOWS] + [f"rstd_{w}" for w in DEFAULT_ROLLING_WINDOWS]
    CALENDAR_COLS = ["dayofweek", "dayofmonth", "weekofyear", "month", "year", "is_weekend",
                     "has_event_name_1", "has_event_name_2"]
    SNAP_COLS = ["snap"]  # Historical flag at origin; not SNAP at the future target.
    PRICE_COLS = ["last_known_price", "price_change_ratio"]
    STATIC_COLS = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]

    def __init__(self, lags=None, rolling_windows=None, rolling_min_periods=DEFAULT_ROLLING_MIN_PERIODS):
        self.lags = list(DEFAULT_LAGS if lags is None else lags)
        self.rolling_windows = list(DEFAULT_ROLLING_WINDOWS if rolling_windows is None else rolling_windows)
        if not self.lags or any(not isinstance(x, int) or isinstance(x, bool) or x < 1
                                for x in self.lags + self.rolling_windows):
            raise ValueError("Lags must be nonempty and lag/window sizes must be positive integers")
        if len(set(self.lags)) != len(self.lags) or len(set(self.rolling_windows)) != len(self.rolling_windows):
            raise ValueError("Lag and rolling window sizes must be distinct")
        if not isinstance(rolling_min_periods, int) or rolling_min_periods < 1:
            raise ValueError("rolling_min_periods must be positive")
        self.rolling_min_periods = rolling_min_periods
        self._encoder = CategoryEncoder()
        self._origin_snapshots = {}
        self._recent_sales = {}
        self._fitted_origin = None

    @property
    def feature_columns(self) -> list[str]:
        return ([f"lag_{lag}" for lag in self.lags]
                + [f"rmean_{w}" for w in self.rolling_windows]
                + [f"rstd_{w}" for w in self.rolling_windows]
                + self.CALENDAR_COLS + self.SNAP_COLS + self.PRICE_COLS + self.STATIC_COLS)

    @staticmethod
    def _validate_history(history: pd.DataFrame, origin: int) -> None:
        if history.empty or history[["series_id", "day_num"]].isna().any().any():
            raise ValueError("History needs nonmissing series/day keys")
        if history.day_num.gt(origin).any() or history.day_num.lt(1).any():
            raise ValueError("History must stop at the forecast origin")
        if history.duplicated(["series_id", "day_num"]).any():
            raise ValueError("Duplicate historical series/day keys")
        for _, group in history.groupby("series_id", observed=True):
            if not np.array_equal(np.sort(group.day_num.to_numpy()), np.arange(1, origin + 1)):
                raise ValueError("History must contain every calendar day through the origin")
        observed = history.units_sold.dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all() or (observed < 0).any():
            raise ValueError("Observed sales must be finite and nonnegative")

    def fit(self, inputs: ForecastInputs) -> FeatureEngineer:
        self._fitted_origin = None
        history = inputs.history.copy()
        self._validate_history(history, inputs.origin)
        self._encoder.fit(history, self.STATIC_COLS)
        self._origin_snapshots = {}
        for sid, group in history.groupby("series_id", observed=True):
            group = group.sort_values("day_num")
            snapshot = _compute_series_origin_snapshot(group, inputs.origin, self.lags,
                                                       self.rolling_windows, self.rolling_min_periods)
            snapshot.update(_price_snapshot_at_origin(group, inputs.origin))
            snapshot["snap"] = float(group.snap.iloc[-1]) if "snap" in group else np.nan
            self._origin_snapshots[str(sid)] = snapshot
        recent = history.loc[history.day_num.ge(inputs.origin - max(self.lags)),
                             ["series_id", "day_num", "units_sold"]]
        self._recent_sales = {(str(sid), int(day)): float(value) if pd.notna(value) else np.nan
                              for sid, day, value in recent.itertuples(index=False, name=None)}
        self._fitted_origin = inputs.origin
        return self

    def _training_table(self, inputs: ForecastInputs) -> pd.DataFrame:
        if self._fitted_origin is None or self._fitted_origin != inputs.origin:
            raise RuntimeError("Call fit() for the current origin before building features")
        self._validate_history(inputs.history, inputs.origin)
        history = inputs.history.sort_values(["series_id", "day_num"]).reset_index(drop=True)
        pieces = [_compute_series_features(group, self.lags, self.rolling_windows, self.rolling_min_periods)
                  for _, group in history.groupby("series_id", observed=True)]
        featured = _encode_calendar(_price_features_train(pd.concat(pieces, ignore_index=True)))
        featured["snap"] = (featured.groupby("series_id", observed=True).snap.shift(1)
                            if "snap" in featured else np.nan)
        featured = self._encoder.transform(featured)
        for column in self.feature_columns:
            if column not in featured:
                featured[column] = np.nan
        return featured

    def build_train_features(self, inputs: ForecastInputs) -> tuple[pd.DataFrame, pd.Series]:
        """Backward-compatible one-step features: sales/price/SNAP stop at t-1."""
        featured = self._training_table(inputs)
        featured = featured.loc[featured.day_num.gt(min(self.lags))].reset_index(drop=True)
        return featured[self.feature_columns].copy(), featured.units_sold.copy()

    def build_direct_train_features(self, inputs: ForecastInputs) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Direct training blocks from origins T-H, T-2H, ... >=1.

        Blocks are disjoint, ending at the fold's origin T. Each label is used
        once, with observed horizons 1..H, matching the development cadence and
        weekday of the prediction origin. Short histories use a partial block
        from day 1. Later historical sales serve only as supervised labels.
        """
        featured = self._training_table(inputs)
        horizon = inputs.horizon
        if horizon < 1 or inputs.origin < 2:
            raise ValueError("Direct training needs a positive horizon and two historical days")
        origins = list(reversed(range(inputs.origin - horizon, 0, -horizon))) or [1]
        assignment = {day: origin for origin in origins
                      for day in range(origin + 1, min(origin + horizon, inputs.origin) + 1)}
        target = featured.loc[featured.day_num.isin(assignment)].copy()
        target["origin"] = target.day_num.map(assignment).astype("int32")
        target["horizon"] = (target.day_num - target.origin).astype("int16")
        snapshots_cols = ([f"rmean_{w}" for w in self.rolling_windows]
                          + [f"rstd_{w}" for w in self.rolling_windows] + self.PRICE_COLS + self.SNAP_COLS)
        snapshots = featured.loc[featured.day_num.isin([origin + 1 for origin in origins]),
                                  ["series_id", "day_num", *snapshots_cols]].copy()
        snapshots["origin"] = snapshots.pop("day_num") - 1
        target = target.drop(columns=snapshots_cols).merge(snapshots, on=["series_id", "origin"],
                                                          how="left", validate="many_to_one", sort=False)
        sales = inputs.history.assign(series_id=inputs.history.series_id.astype(str)).set_index(
            ["series_id", "day_num"]).units_sold
        for lag in self.lags:
            required = target.day_num - lag
            keys = pd.MultiIndex.from_arrays([target.series_id.astype(str), required])
            values = sales.reindex(keys).to_numpy(dtype=float, na_value=np.nan, copy=True)
            values[required.to_numpy() > target.origin.to_numpy()] = np.nan
            target[f"lag_{lag}"] = values
        target = target.rename(columns={"day_num": "target_day_num", "date": "target_date"})
        target = target.loc[target.units_sold.notna()].reset_index(drop=True)
        if target.empty:
            raise ValueError("No observed labels available for direct training")
        return (target[[*self.feature_columns, "horizon"]].copy(), target.units_sold.copy(),
                target[FUTURE_KEYS].copy())

    def build_predict_features(self, inputs: ForecastInputs) -> pd.DataFrame:
        if self._fitted_origin is None or self._fitted_origin != inputs.origin:
            raise RuntimeError("Call fit() for the current origin before predicting")
        columns = FUTURE_KEYS + [col for col in KNOWN_CALENDAR if col in inputs.future_known]
        future = inputs.future_known[columns].copy()
        if set(future.series_id.astype(str)) - self._origin_snapshots.keys():
            raise ValueError("Requested series missing from fitted history")
        horizons = future.horizon.to_numpy(dtype=float)
        if (not future.origin.eq(inputs.origin).all() or not np.isfinite(horizons).all()
                or not np.equal(horizons, np.floor(horizons)).all()
                or not future.target_day_num.eq(inputs.origin + future.horizon).all()
                or not future.horizon.between(1, inputs.horizon).all()):
            raise ValueError("Invalid future origin/horizon grid")
        future["_row_position"] = np.arange(len(future))
        future = _encode_calendar(future, date_col="target_date")
        future = future.merge(inputs.static[["series_id", *self.STATIC_COLS]], on="series_id",
                              how="left", validate="many_to_one", sort=False)
        future = self._encoder.transform(future)
        rows = []
        for sid, snapshot in self._origin_snapshots.items():
            row = {"series_id": sid}
            for window in self.rolling_windows:
                row[f"rmean_{window}"] = snapshot[f"origin_rmean_{window}"]
                row[f"rstd_{window}"] = snapshot[f"origin_rstd_{window}"]
            row.update({col: snapshot[col] for col in [*self.PRICE_COLS, "snap"]})
            rows.append(row)
        future = future.merge(pd.DataFrame(rows), on="series_id", how="left", validate="many_to_one", sort=False)
        for lag in self.lags:
            required = future.target_day_num.to_numpy() - lag
            future[f"lag_{lag}"] = [self._recent_sales.get((str(sid), int(day)), np.nan)
                                    if day <= inputs.origin else np.nan
                                    for sid, day in zip(future.series_id, required)]
        for column in self.feature_columns:
            if column not in future:
                future[column] = np.nan
        return future.sort_values("_row_position")[self.feature_columns].reset_index(drop=True)

    def to_state(self) -> dict:
        """JSON-safe encoder and complete frozen inference state."""
        if self._fitted_origin is None:
            raise RuntimeError("Call fit() before saving feature state")
        def finite(value):
            return float(value) if np.isfinite(value) else None
        return {"version": FEATURE_VERSION, "origin": int(self._fitted_origin), "lags": self.lags,
                "rolling_windows": self.rolling_windows, "rolling_min_periods": self.rolling_min_periods,
                "feature_columns": self.feature_columns, "mappings": self._encoder.mappings_,
                "snapshots": {sid: {key: finite(value) for key, value in row.items()}
                              for sid, row in self._origin_snapshots.items()},
                "recent_sales": [[sid, day, finite(value)] for (sid, day), value in self._recent_sales.items()]}

    @classmethod
    def from_state(cls, state: dict) -> FeatureEngineer:
        if state.get("version") != FEATURE_VERSION:
            raise ValueError("Unsupported feature-state version")
        builder = cls(state["lags"], state["rolling_windows"], state["rolling_min_periods"])
        if builder.feature_columns != state["feature_columns"]:
            raise ValueError("Saved feature schema differs from the implementation")
        builder._encoder.mappings_ = state["mappings"]
        builder._origin_snapshots = {sid: {key: np.nan if value is None else value for key, value in row.items()}
                                     for sid, row in state["snapshots"].items()}
        builder._recent_sales = {(sid, int(day)): np.nan if value is None else value
                                 for sid, day, value in state["recent_sales"]}
        builder._fitted_origin = int(state["origin"])
        return builder


def assert_no_future_leakage(X, history: pd.DataFrame, origin: int) -> None:
    """Reject histories outside the fitting boundary."""
    if history.day_num.max() > origin:
        raise AssertionError(f"History contains days after origin {origin}: leakage")


def verify_feature_availability(feature_df: pd.DataFrame, origin: int, horizon: int, mode="predict") -> None:
    """A target-relative lag is unavailable when its horizon exceeds that lag."""
    if mode != "predict" or "horizon" not in feature_df:
        return
    for column in feature_df:
        if column.startswith("lag_") and column[4:].isdigit():
            lag = int(column[4:])
            if feature_df.loc[feature_df.horizon.gt(lag), column].notna().any():
                raise AssertionError(f"{column}: future sales leakage at horizon > lag")


def build_feature_datasets(config_path="experiment.yaml", output_dir="data/processed/features") -> dict[str, Any]:
    """Export the actual direct-model training and prediction matrices plus keys."""
    from src.config import ROOT, config_hash, load_config
    from src.data.preprocess import load_panel
    from src.evaluation.backtesting import construct_inputs
    from src.evaluation.splits import development_splits
    cfg = load_config(config_path)
    panel = load_panel(cfg, cutoff=cfg["forecast"]["final_origin"])
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)
    folds = development_splits(cfg)
    metadata = {"feature_version": FEATURE_VERSION, "strategy": TRAINING_STRATEGY,
                "config_path": str(config_path), "config_sha256": config_hash(cfg),
                "cutoff": cfg["forecast"]["final_origin"], "n_folds": len(folds),
                "origins": [fold.origin for fold in folds], "horizon": cfg["forecast"]["horizon"], "folds": {}}
    for fold in folds:
        inputs = construct_inputs(panel, fold)
        builder = FeatureEngineer().fit(inputs)
        X, y, keys = builder.build_direct_train_features(inputs)
        predict = builder.build_predict_features(inputs)
        assert_no_future_leakage(X, inputs.history, inputs.origin)
        verify_feature_availability(X, inputs.origin, inputs.horizon)
        train = X.copy()
        for key in FUTURE_KEYS:
            train[key] = keys[key].to_numpy()
        train["target"] = y.to_numpy()
        for key in FUTURE_KEYS:
            predict[key] = inputs.future_known[key].to_numpy()
        verify_feature_availability(predict, inputs.origin, inputs.horizon)
        train_path = destination / f"train_features_origin_{fold.origin}.parquet"
        predict_path = destination / f"predict_features_origin_{fold.origin}.parquet"
        train.to_parquet(train_path, index=False)
        predict.to_parquet(predict_path, index=False)
        metadata["folds"][str(fold.origin)] = {
            "origin": fold.origin, "training_origins": sorted(keys.origin.unique().tolist()),
            "train_rows": len(train), "train_cols": len(train.columns),
            "predict_rows": len(predict), "predict_cols": len(predict.columns),
            "train_file": str(train_path), "predict_file": str(predict_path)}
        print(f"Origin {fold.origin}: train {train.shape}; predict {predict.shape}", flush=True)
    metadata.update(feature_columns=[*builder.feature_columns, "horizon"], lags=builder.lags,
                    rolling_windows=builder.rolling_windows)
    (destination / "feature_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build origin-safe direct-horizon features")
    parser.add_argument("--config", default="experiment.yaml")
    parser.add_argument("--output-dir", default="data/processed/features")
    args = parser.parse_args()
    build_feature_datasets(args.config, args.output_dir)
