# Milestone 8: Leakage-Safe Feature Engineering

## 1. Objective and Overview

Milestone 8 converts historical demand and known future calendar covariates into supervised learning feature matrices for tree-based gradient boosting models (Milestone 9: LightGBM).

Every feature is engineered under a strict **leakage-safe** contract:
1. **Training features** are computed per-series exclusively from observations preceding each target day ($t < \text{target\_day}$).
2. **Prediction features** for the 28-day forecast horizon ($h = 1, \dots, 28$) use exclusively information available at or before the forecast origin $T$. Unknown future sales ($h > \text{lag}$) are represented as `NaN`, which tree-based gradient boosters handle natively via missing-value splits.
3. **No future actuals, future prices, future SNAP, or quality flags** are ever ingested as predictor inputs.
4. **Categorical encodings and origin snapshots** are fitted fold-locally inside `fit()` on historical rows up to the fold's origin.

---

## 2. Feature Definitions

The feature matrix contains 26 predictor columns plus the target column (`units_sold` / `target`):

| Feature Category | Column Name(s) | Computation & Source | Leakage-Safety Rule |
| :--- | :--- | :--- | :--- |
| **Demand Lags** | `lag_1`, `lag_7`, `lag_14`, `lag_28` | Units sold at $(t - k)$ days. | In prediction, for horizon $h$, available only when $h \le k$ (yielding observation at origin $T - (k - h)$). For $h > k$, value is `NaN`. |
| **Rolling Means** | `rmean_7`, `rmean_14`, `rmean_28` | Trailing mean of sales over 7, 14, 28 days preceding $t$. | In prediction, fixed snapshot of trailing window ending at origin $T$. |
| **Rolling Std Devs** | `rstd_7`, `rstd_14`, `rstd_28` | Trailing standard deviation over 7, 14, 28 days preceding $t$. | In prediction, fixed snapshot of trailing window ending at origin $T$. |
| **Calendar: Day & Week** | `dayofweek`, `dayofmonth`, `weekofyear`, `is_weekend` | Derived from ISO calendar date (`date` / `target_date`). | Future dates are pre-announced and known ahead. |
| **Calendar: Month & Year** | `month`, `year` | Month (1–12) and Year integers. | Future dates are pre-announced and known ahead. |
| **Event Indicators** | `has_event_name_1`, `has_event_name_2` | Binary flag indicating presence of named cultural/sporting event. | Known future calendar events from `calendar.csv`. |
| **SNAP Benefit Days** | `snap` | State-specific SNAP eligibility indicator (0 or 1). | Conservatively set to 0 in prediction per experiment protocol. |
| **Historical Price** | `last_known_price` | Most recent known shelf price forward-filled within series. | Forward-filled up to $t$ (train) or origin $T$ (predict). |
| **Price Dynamics** | `price_change_ratio` | Ratio of current shelf price to price 7 days earlier ($p_t / p_{t-7}$). | Evaluated at $t$ (train) or origin $T$ (predict). |
| **Static Identifiers** | `item_id`, `store_id`, `dept_id`, `cat_id`, `state_id` | Integer-encoded categorical mappings learned on fold history. | Fitted on fold history; unseen levels map to -1. |

---

## 3. Multi-Step Prediction Strategy and Leakage Safeguards

For multi-step forecasting across 28 future horizons, two distinct classes of covariates exist:

1. **Known Future Covariates**:
   - Calendar properties (day-of-week, day-of-month, month, year, weekend status) and scheduled event flags are deterministic calendar functions known for all 28 target days.
2. **Autoregressive / Past-Only Covariates**:
   - For target day $T + h$ and lag offset $k$:
     $$\text{required\_day} = (T + h) - k = T - (k - h)$$
     - When $h \le k$, $\text{required\_day} \le T$, which is historical and observed.
     - When $h > k$, $\text{required\_day} > T$, which requires future observations that do not exist at origin $T$. These positions are filled with `NaN`.

### Null Distribution in Prediction Features

Across the 6,720 prediction rows per fold (240 series $\times$ 28 horizons):

| Feature | Non-Null Horizons | Null Fraction | Explanation |
| :--- | :--- | :---: | :--- |
| `lag_1` | Horizon 1 only | **96.4% (27/28)** | Lags 1 day back; known only for step 1 ($T + 1 - 1 = T$). |
| `lag_7` | Horizons 1–7 | **75.0% (21/28)** | Lags 7 days back; known for all steps up to 1 full week. |
| `lag_14` | Horizons 1–14 | **50.0% (14/28)** | Lags 14 days back; known for steps 1–14 (first 2 weeks). |
| `lag_28` | Horizons 1–28 | **0.0% (0/28)** | Lags 28 days back; known for all 28 steps in the forecast horizon. |
| `rmean_*`, `rstd_*` | Horizons 1–28 | **0.0%** | Origin snapshot trailing statistics known for all steps. |
| Calendar & Static | Horizons 1–28 | **0.0%** | Known ahead or statically invariant. |

Tree-based algorithms (such as LightGBM) natively support missing values by choosing optimal default split directions for NaNs during training.

---

## 4. Generated Datasets and Metadata

Running `python -m src.features.feature_engineering` (or `python -m scripts.build_features`) produces the following validated datasets in `data/processed/features/`:

| File | Rows | Columns | Description |
| :--- | ---: | ---: | :--- |
| `train_features_origin_1801.parquet` | 432,000 | 27 | Fold 1 training matrix ($d_2 \dots d_{1801}$) with target. |
| `predict_features_origin_1801.parquet` | 6,720 | 30 | Fold 1 prediction grid ($h=1 \dots 28$) with keys. |
| `train_features_origin_1829.parquet` | 438,720 | 27 | Fold 2 training matrix ($d_2 \dots d_{1829}$) with target. |
| `predict_features_origin_1829.parquet` | 6,720 | 30 | Fold 2 prediction grid ($h=1 \dots 28$) with keys. |
| `train_features_origin_1857.parquet` | 445,440 | 27 | Fold 3 training matrix ($d_2 \dots d_{1857}$) with target. |
| `predict_features_origin_1857.parquet` | 6,720 | 30 | Fold 3 prediction grid ($h=1 \dots 28$) with keys. |
| `train_features_origin_1885.parquet` | 452,160 | 27 | Fold 4 training matrix ($d_2 \dots d_{1885}$) with target. |
| `predict_features_origin_1885.parquet` | 6,720 | 30 | Fold 4 prediction grid ($h=1 \dots 28$) with keys. |
| `feature_metadata.json` | — | — | Provenance, column schemas, and fold record counts. |

---

## 5. Verification and Automated Tests

The implementation is verified by 28 unit and integration tests in `tests/test_feature_engineering.py`:

- **Leakage Prevention**: Tests mutate future sales, prices, and event columns; feature values and predictions remain completely unaffected.
- **Horizon-Lag Boundary**: Tests verify that for $h > k$, lag features are strictly `NaN`, and for $h \le k$, lag features are correctly populated from past observations.
- **Categorical Mapping**: Tests verify consistent integer mapping, out-of-vocabulary handling (mapped to -1), and fold isolation.
- **Dataset Generation**: Automated end-to-end verification of parquet serialization and metadata creation across all development folds.

To reproduce:
```bash
# Run unit and integration tests
python -m pytest tests/test_feature_engineering.py -v

# Generate feature datasets for all development folds
python -m scripts.build_features
```
