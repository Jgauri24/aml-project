"""Dataset construction, categorical encoding, covariate extraction, and validation for DeepAR.

Ensures strict chronological isolation: training instances only use observations
available strictly before the forecast origin, and validation uses a held-out historical
chronological window. No future actuals or forbidden features leak into any window.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.load_data import STATIC
from src.evaluation.backtesting import ForecastInputs
from src.models.deepar_config import DeepARConfig

logger = logging.getLogger(__name__)


def handle_missing_targets(
    y: np.ndarray,
    policy: str = "zero",
) -> np.ndarray:
    """Handle missing observations (NaN / inf) in target series according to specified policy.

    Policies:
    - "zero": Unobserved sales set to 0.0 (canonical assumption for retail intermittent demand).
    - "forward_fill": Forward-fill last observed value, with remaining leading NaNs filled with 0.0.
    - "mean": Fill with the mean of observed non-missing values (or 0.0 if all missing).
    """
    y_clean = np.asarray(y, dtype=np.float32).copy()
    mask_missing = ~np.isfinite(y_clean)
    if not mask_missing.any():
        return y_clean

    if policy == "zero":
        y_clean[mask_missing] = 0.0
    elif policy == "forward_fill":
        s = pd.Series(y_clean)
        s = s.ffill().fillna(0.0)
        y_clean = s.to_numpy(dtype=np.float32)
    elif policy == "mean":
        observed = y_clean[~mask_missing]
        mean_val = float(np.mean(observed)) if len(observed) > 0 else 0.0
        y_clean[mask_missing] = mean_val
    else:
        raise ValueError(f"Unknown missing target policy: {policy}")

    return y_clean


def validate_deepar_inputs(inputs: ForecastInputs) -> None:
    """Validate data integrity for DeepAR training and inference."""
    if inputs.history is None or inputs.history.empty:
        raise ValueError("ForecastInputs history cannot be empty")
    if inputs.static is None or inputs.static.empty:
        raise ValueError("ForecastInputs static metadata cannot be empty")
    if inputs.future_known is None or inputs.future_known.empty:
        raise ValueError("ForecastInputs future_known cannot be empty")

    # Check origin day boundary
    max_history_day = inputs.history["day_num"].max()
    if max_history_day > inputs.origin:
        raise ValueError(
            f"History contains day_num {max_history_day} exceeding origin {inputs.origin}"
        )

    # Check units sold nonnegativity and validity
    sales = inputs.history["units_sold"].dropna()
    if (sales < 0).any():
        raise ValueError("Historical units_sold contains negative values")
    if np.isinf(sales.to_numpy(dtype=float)).any():
        raise ValueError("Historical units_sold contains non-finite values (inf / -inf)")

    # Check series alignment
    history_series = set(inputs.history["series_id"].unique())
    static_series = set(inputs.static["series_id"].unique())
    future_series = set(inputs.future_known["series_id"].unique())

    if history_series != static_series:
        missing = static_series - history_series
        raise ValueError(f"Series mismatch between history and static metadata: {len(missing)} missing")
    if history_series != future_series:
        missing = future_series - history_series
        raise ValueError(f"Series mismatch between history and future_known: {len(missing)} missing")


class StaticCategoryEncoder:
    """Encodes static categorical features with index 0 reserved for unknown (<unk>) categories."""

    def __init__(self, cat_columns: List[str] | None = None) -> None:
        self.cat_columns = cat_columns or STATIC
        self.mappings_: Dict[str, Dict[str, int]] = {}
        self.vocab_sizes_: Dict[str, int] = {}

    def fit(self, static_df: pd.DataFrame) -> StaticCategoryEncoder:
        """Fit vocabulary mappings from static dataframe."""
        self.mappings_ = {}
        self.vocab_sizes_ = {}
        for col in self.cat_columns:
            if col not in static_df.columns:
                raise ValueError(f"Required static categorical column '{col}' missing from static dataframe")
            valid_vals = static_df[col].dropna().astype(str).unique()
            unique_vals = sorted(v for v in valid_vals if v not in ["nan", "None", "<unk>"])
            # index 0 is reserved for unknown categories (<unk>)
            mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            self.mappings_[col] = mapping
            self.vocab_sizes_[col] = len(unique_vals) + 1  # include <unk>
        return self

    def transform_series(self, static_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Transform static dataframe into integer encoded arrays indexed by series_id."""
        result = {}
        for col in self.cat_columns:
            mapping = self.mappings_[col]
            encoded = static_df[col].apply(
                lambda x: mapping.get(str(x), 0) if pd.notna(x) and str(x) not in ["nan", "None"] else 0
            ).to_numpy(dtype=np.int64)
            result[col] = encoded
        return result


def extract_dynamic_covariates(df: pd.DataFrame) -> np.ndarray:
    """Extract known-future dynamic calendar covariates from dataframe.

    Features generated:
    - wday sine & cosine (weekly cyclicality)
    - month sine & cosine (annual cyclicality)
    - is_weekend (binary indicator for Saturday/Sunday)
    - has_event_1 (binary indicator for primary event)
    - has_event_2 (binary indicator for secondary event)
    - day_progress (linear normalized day index (day_num - 1) / 1941.0)
    """
    n_rows = len(df)
    features = []

    # 1. wday cyclical features (wday is 1..7)
    if "wday" in df.columns:
        wday = df["wday"].to_numpy(dtype=np.float32)
        wday_sin = np.sin(2.0 * np.pi * (wday - 1.0) / 7.0)
        wday_cos = np.cos(2.0 * np.pi * (wday - 1.0) / 7.0)
        features.extend([wday_sin, wday_cos])
    else:
        features.extend([np.zeros(n_rows, dtype=np.float32), np.zeros(n_rows, dtype=np.float32)])

    # 2. month cyclical features (month is 1..12)
    if "month" in df.columns:
        month = df["month"].to_numpy(dtype=np.float32)
        month_sin = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
        month_cos = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
        features.extend([month_sin, month_cos])
    else:
        features.extend([np.zeros(n_rows, dtype=np.float32), np.zeros(n_rows, dtype=np.float32)])

    # 3. is_weekend indicator (in M5: wday 1=Saturday, wday 2=Sunday)
    if "wday" in df.columns:
        is_weekend = np.isin(df["wday"], [1, 2]).astype(np.float32)
        features.append(is_weekend)
    elif "weekday" in df.columns:
        is_weekend = df["weekday"].isin(["Saturday", "Sunday"]).to_numpy(dtype=np.float32)
        features.append(is_weekend)
    else:
        features.append(np.zeros(n_rows, dtype=np.float32))

    # 4. event indicators
    for ev_col in ["event_name_1", "event_name_2"]:
        if ev_col in df.columns:
            has_ev = (df[ev_col].notna() & ~df[ev_col].astype(str).isin(["nan", "None", ""])).to_numpy(dtype=np.float32)
            features.append(has_ev)
        else:
            features.append(np.zeros(n_rows, dtype=np.float32))

    # 5. normalized day progress
    day_col = "day_num" if "day_num" in df.columns else ("target_day_num" if "target_day_num" in df.columns else None)
    if day_col is not None:
        day_prog = (df[day_col].to_numpy(dtype=np.float32) - 1.0) / 1941.0
        features.append(day_prog)
    else:
        features.append(np.zeros(n_rows, dtype=np.float32))

    # Stack along axis 1: shape (N, num_features)
    cov_matrix = np.column_stack(features).astype(np.float32)
    cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return cov_matrix


class DeepARTrainingDataset(Dataset):
    """PyTorch Dataset yielding rolling chronological context + prediction windows for DeepAR training."""

    def __init__(
        self,
        targets: List[np.ndarray],
        covariates: List[np.ndarray],
        static_cats: np.ndarray,
        scales: np.ndarray,
        context_length: int,
        prediction_length: int,
        windows_per_series: int = 8,
        is_validation: bool = False,
        seed: int = 42,
    ) -> None:
        self.targets = targets              # list of 1D arrays per series
        self.covariates = covariates        # list of 2D arrays per series
        self.static_cats = static_cats      # 2D array (N_series, N_static_cats)
        self.scales = scales                # 1D array (N_series,)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_length = context_length + prediction_length
        self.num_series = len(targets)
        self.is_validation = is_validation

        self.samples_: List[Tuple[int, int]] = []  # list of (series_idx, start_pos)
        self._build_samples(windows_per_series, seed)

    def _build_samples(self, windows_per_series: int, seed: int) -> None:
        rng = np.random.RandomState(seed)
        self.samples_ = []

        for s_idx in range(self.num_series):
            series_len = len(self.targets[s_idx])

            if self.is_validation:
                # Validation samples use the final window ending at series_len
                start = max(0, series_len - self.window_length)
                self.samples_.append((s_idx, start))
            else:
                max_start = series_len - self.window_length
                if max_start <= 0:
                    self.samples_.append((s_idx, 0))
                else:
                    n_windows = min(windows_per_series, max_start + 1)
                    starts = rng.choice(max_start + 1, size=n_windows, replace=False)
                    for start in sorted(starts):
                        self.samples_.append((s_idx, int(start)))

    def __len__(self) -> int:
        return len(self.samples_)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s_idx, start = self.samples_[idx]
        target_seq = self.targets[s_idx]
        cov_seq = self.covariates[s_idx]
        series_len = len(target_seq)

        end = start + self.window_length
        target = target_seq[start:end]
        cov = cov_seq[start:end]

        # If series is shorter than window_length, pad on the left
        if len(target) < self.window_length:
            pad_len = self.window_length - len(target)
            target = np.pad(target, (pad_len, 0), mode="constant", constant_values=0.0)
            if cov.shape[1] > 0:
                first_cov = cov[0:1] if len(cov) > 0 else np.zeros((1, cov.shape[1]), dtype=np.float32)
                cov_pad = np.repeat(first_cov, pad_len, axis=0)
                cov = np.vstack([cov_pad, cov])
            else:
                cov = np.zeros((self.window_length, 0), dtype=np.float32)

        static_cat = self.static_cats[s_idx]
        scale = self.scales[s_idx]

        # In case a series has zero sales across history, ensure scale >= 1.0
        if scale <= 0.0 or not np.isfinite(scale):
            scale = 1.0

        return {
            "target": torch.from_numpy(target).float(),
            "covariates": torch.from_numpy(cov).float(),
            "static_cat": torch.from_numpy(static_cat).long(),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "series_idx": torch.tensor(s_idx, dtype=torch.long),
        }


class DeepARPreparedData:
    """Holds processed arrays, encoders, and dataset builders for a given ForecastInputs origin."""

    def __init__(
        self,
        config: DeepARConfig,
        encoder: StaticCategoryEncoder,
        series_ids: List[str],
        train_dataset: DeepARTrainingDataset,
        val_dataset: DeepARTrainingDataset,
        inference_data: Dict[str, Any],
        covariate_dim: int,
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.series_ids = series_ids
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.inference_data = inference_data
        self.covariate_dim = covariate_dim


def prepare_deepar_data(inputs: ForecastInputs, config: DeepARConfig) -> DeepARPreparedData:
    """Transform ForecastInputs into model-ready training, validation, and inference datasets."""
    validate_deepar_inputs(inputs)

    # 1. Fit categorical encoder on static metadata
    encoder = StaticCategoryEncoder()
    encoder.fit(inputs.static)

    # 2. Extract series list ordered by static dataframe
    series_df = inputs.static[["series_id", *STATIC]].drop_duplicates().sort_values("series_id").reset_index(drop=True)
    series_ids = series_df["series_id"].tolist()
    n_series = len(series_ids)

    # Encode static categories as 2D array (N_series, N_cats)
    encoded_cats_dict = encoder.transform_series(series_df)
    static_cats = np.column_stack([encoded_cats_dict[col] for col in encoder.cat_columns])

    # 3. Process time-series history
    # Ensure chronological sort per series
    hist_sorted = inputs.history.sort_values(["series_id", "day_num"]).reset_index(drop=True)

    # Compute dynamic covariates across the full history
    covariates_full = extract_dynamic_covariates(hist_sorted)
    cov_dim = covariates_full.shape[1] if config.use_calendar_features else 0

    # Group history by series_id
    targets_train: List[np.ndarray] = []
    targets_val: List[np.ndarray] = []
    covariates_train: List[np.ndarray] = []
    covariates_val: List[np.ndarray] = []
    scales: List[float] = []

    # Map series_id to group slices
    hist_grouped = {sid: grp for sid, grp in hist_sorted.groupby("series_id", observed=True)}

    # For chronological validation: hold out the final val_window_length days
    val_len = config.val_window_length
    policy = getattr(config, "missing_target_policy", "zero")
    missing_count = 0
    total_obs = 0

    for s_idx, sid in enumerate(series_ids):
        grp = hist_grouped[sid]
        raw_y = grp["units_sold"].to_numpy(dtype=np.float32)
        total_obs += len(raw_y)
        missing_count += int((~np.isfinite(raw_y)).sum())

        # Clean missing observations according to configured policy
        y = handle_missing_targets(raw_y, policy=policy)

        # Scale factor nu_i = 1.0 + (1 / T) * sum(y_t)
        # Computed strictly on training portion ignoring NaNs to prevent any leakage
        raw_y_train = raw_y[:-val_len] if len(raw_y) > val_len + config.context_length else raw_y
        valid_train = raw_y_train[np.isfinite(raw_y_train)]
        mean_val = float(np.mean(valid_train)) if len(valid_train) > 0 else 0.0
        scale = 1.0 + max(0.0, mean_val) if config.scaling else 1.0
        scales.append(scale)

        if config.use_calendar_features:
            cov = extract_dynamic_covariates(grp)
        else:
            cov = np.zeros((len(grp), 0), dtype=np.float32)

        # Training history: up to (len - val_len)
        targets_train.append(y[:-val_len] if len(y) > val_len + config.context_length else y)
        covariates_train.append(cov[:-val_len] if len(cov) > val_len + config.context_length else cov)

        # Validation history: full history up to origin
        targets_val.append(y)
        covariates_val.append(cov)

    if missing_count > 0:
        logger.info(
            f"Handled {missing_count} missing/non-finite observations ({100.0 * missing_count / total_obs:.2f}%) "
            f"using policy '{policy}'"
        )

    scales_arr = np.array(scales, dtype=np.float32)

    train_ds = DeepARTrainingDataset(
        targets=targets_train,
        covariates=covariates_train,
        static_cats=static_cats,
        scales=scales_arr,
        context_length=config.context_length,
        prediction_length=config.prediction_length,
        windows_per_series=config.windows_per_series_train,
        is_validation=False,
        seed=config.seed,
    )

    val_ds = DeepARTrainingDataset(
        targets=targets_val,
        covariates=covariates_val,
        static_cats=static_cats,
        scales=scales_arr,
        context_length=config.context_length,
        prediction_length=config.prediction_length,
        windows_per_series=1,
        is_validation=True,
        seed=config.seed,
    )

    # 4. Prepare inference inputs
    # For inference, context window is the last context_length days up to origin
    context_targets = []
    context_covs = []
    future_covs = []

    # Prepare future covariates from future_known
    future_sorted = inputs.future_known.sort_values(["series_id", "horizon"]).reset_index(drop=True)
    future_grouped = {sid: grp for sid, grp in future_sorted.groupby("series_id", observed=True)}

    for s_idx, sid in enumerate(series_ids):
        grp_hist = hist_grouped[sid]
        raw_y_hist = grp_hist["units_sold"].to_numpy(dtype=np.float32)
        y_hist = handle_missing_targets(raw_y_hist, policy=policy)
        # Context window: last context_length days up to origin
        if len(y_hist) < config.context_length:
            pad_len = config.context_length - len(y_hist)
            y_ctx = np.pad(y_hist, (pad_len, 0), mode="constant", constant_values=0.0)
            if config.use_calendar_features:
                cov_hist = extract_dynamic_covariates(grp_hist)
                first_cov = cov_hist[0:1] if len(cov_hist) > 0 else np.zeros((1, cov_hist.shape[1]), dtype=np.float32)
                cov_pad = np.repeat(first_cov, pad_len, axis=0)
                cov_ctx = np.vstack([cov_pad, cov_hist])
            else:
                cov_ctx = np.zeros((config.context_length, 0), dtype=np.float32)
        else:
            y_ctx = y_hist[-config.context_length:]
            if config.use_calendar_features:
                cov_hist = extract_dynamic_covariates(grp_hist)
                cov_ctx = cov_hist[-config.context_length:]
            else:
                cov_ctx = np.zeros((config.context_length, 0), dtype=np.float32)

        if config.use_calendar_features:
            grp_future = future_grouped[sid]
            cov_fut = extract_dynamic_covariates(grp_future)
        else:
            cov_fut = np.zeros((inputs.horizon, 0), dtype=np.float32)

        context_targets.append(y_ctx)
        context_covs.append(cov_ctx)
        future_covs.append(cov_fut)

    inference_data = {
        "context_targets": torch.from_numpy(np.stack(context_targets)).float(),  # (N_series, context_len)
        "context_covs": torch.from_numpy(np.stack(context_covs)).float(),        # (N_series, context_len, cov_dim)
        "future_covs": torch.from_numpy(np.stack(future_covs)).float(),          # (N_series, horizon, cov_dim)
        "static_cats": torch.from_numpy(static_cats).long(),                     # (N_series, N_cats)
        "scales": torch.from_numpy(scales_arr).float(),                          # (N_series,)
        "series_ids": series_ids,
        "origin": inputs.origin,
        "horizon": inputs.horizon,
    }

    return DeepARPreparedData(
        config=config,
        encoder=encoder,
        series_ids=series_ids,
        train_dataset=train_ds,
        val_dataset=val_ds,
        inference_data=inference_data,
        covariate_dim=cov_dim,
    )
