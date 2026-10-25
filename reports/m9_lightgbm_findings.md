# Milestone 9: Gradient Boosting (LightGBM) Forecaster Findings

**Audit, 2026-10-25: partially complete.** The runner executes, but its training
`horizon` is entirely NaN, trained model files are not saved, and the returned
tuning metadata is not persisted to the execution JSON. The benchmark numbers
below differ from the new audit run and should be treated as historical claims,
not reproduced results. See [milestone verification](milestone_verification.md)
for current measurements and the unresolved feature-contract failures.

## 1. Objective and Architecture

Milestone 9 establishes a global tabular machine learning model using **LightGBM** that learns demand patterns across all 240 series simultaneously.

Key design attributes:
- **Global Estimation**: Unlike the per-series Kalman filters and naive heuristics, a single unified gradient-boosted decision tree model is trained on pooled historical observations across all SKUs and stores.
- **Leakage-Safe Feature Inputs**: Utilizes the Milestone 8 [`FeatureEngineer`](../src/features/feature_engineering.py) pipeline to provide 26 features (demand lags, rolling means and standard deviations, calendar dynamics, event indicators, SNAP flags, historical prices, and fold-fitted categorical encodings).
- **Direct Multi-Horizon Strategy**: Direct multi-step prediction across the 28-day horizon. For prediction steps $h > \text{lag}$, unavailable future sales are represented as `NaN`, which LightGBM branches natively on via optimal missing-value splits.
- **Objective Function**: Trained using the L1 regression objective (`regression_l1` / MAE loss), which aligns directly with median demand and provides robustness to intermittent, sparse demand.

---

## 2. Rolling-Validation Hyperparameter Tuning

Hyperparameter selection was conducted via rolling-window validation on the first three development folds (origins 1801, 1829, 1857), holding out fold 4 (origin 1885) to assess out-of-sample generalization. WAPE was the optimization criterion:

| Candidate Configuration | Validation Mean WAPE (Folds 1–3) | Status |
| :--- | :---: | :---: |
| `num_leaves: 31, learning_rate: 0.05, n_estimators: 500` | 0.8278 | Evaluated |
| `num_leaves: 63, learning_rate: 0.05, n_estimators: 500` | 0.8207 | Evaluated |
| **`num_leaves: 127, learning_rate: 0.03, n_estimators: 700`** | **0.8177** | **Selected Winner** |

The higher-capacity model (`num_leaves=127`) with a smaller learning rate (`0.03`) yielded the lowest validation error, effectively capturing category and store interactions across the pooled series.

---

## 3. Development Backtest Performance

The tuned model was evaluated across all four standard development folds (26,880 total predictions, 6,720 per fold):

| Scope | Origin Day | Calendar Window | MAE | RMSE | WAPE Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | $d_{1801}$ | 2016-01-04 – 2016-01-31 | 0.9697 | 2.5232 | 0.8188 |
| Fold 2 | $d_{1829}$ | 2016-02-01 – 2016-02-28 | 0.9814 | 2.2898 | 0.8139 |
| Fold 3 | $d_{1857}$ | 2016-02-29 – 2016-03-27 | 0.9580 | 2.3058 | 0.8205 |
| Fold 4 | $d_{1885}$ | 2016-03-28 – 2016-04-24 | 1.0406 | 2.4706 | 0.8172 |
| **All Folds Pooled** | — | **Full Development Period** | **0.9874** | **2.3995** | **0.8176** |

All predictions are non-negative and finite. Point-only forecasts yield `NaN` for quantile pinball losses, adhering strictly to the experiment protocol.

---

## 4. Benchmark Comparison across All Milestones

The table below summarizes all models evaluated on the identical 240 series and four development folds under the repository's evaluation contract:

| Model | Model Family | Pooled MAE | Pooled RMSE | Pooled WAPE | Winner vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LightGBM (Tuned)** | **Global GBDT (M9)** | **0.9874** | **2.3995** | **0.8176** | **Overall Best** |
| `kalman_local_level_seasonal` | State-Space Filter (M7) | 1.0007 | 2.0535 | 0.8286 | Best Statistical Model |
| `kalman_local_trend_seasonal` | State-Space Filter (M7) | 1.0030 | 2.0576 | 0.8305 | Competitive |
| `kalman_local_level` | State-Space Filter (M7) | 1.0106 | 2.0613 | 0.8368 | Baseline Filter |
| `kalman_single_sku` | State-Space Filter (M6) | 1.0115 | 2.0776 | 0.8375 | Single-SKU Variant |
| `kalman_local_trend` | State-Space Filter (M7) | 1.0225 | 2.0976 | 0.8466 | Trend Overfitting |
| `naive_weekly_seasonal` | Heuristic Baseline (M5) | 1.1869 | 2.5487 | 0.9828 | Naive Winner |
| `naive_last_value` | Heuristic Baseline (M5) | 1.2479 | 2.5151 | 1.0332 | Simple Baseline |

### Key Takeaways:
1. **LightGBM wins by lowest development WAPE (0.8176)**, outperforming the best statistical baseline (`kalman_local_level_seasonal`, WAPE 0.8286) by 1.1 percentage points and seasonal naive (WAPE 0.9828) by 16.5 percentage points.
2. **Global cross-learning**: By learning joint representations across all 240 series, LightGBM overcomes sparsity issues that limit per-series models on intermittent SKUs.
3. **Execution Efficiency**: Training LightGBM across all 4 folds took **~53 seconds** (including rolling hyperparameter tuning), compared to **>600 seconds** for multi-series Kalman MLE optimization.

---

## 5. Feature Importance Analysis

Feature importance by total split gain averaged across all development folds:

| Rank | Feature Name | Importance (Gain) | Description & Role |
| :---: | :--- | ---: | :--- |
| 1 | `rstd_28` | 428,628 | Trailing 28-day demand volatility. |
| 2 | `rmean_28` | 348,233 | Trailing 28-day demand level. |
| 3 | `rmean_14` | 344,343 | Trailing 2-week baseline level. |
| 4 | `rstd_14` | 319,613 | Trailing 2-week volatility. |
| 5 | `last_known_price` | 172,302 | Shelf price level (proxy for volume tier / product scale). |
| 6 | `lag_1` | 140,715 | Most recent sales observation (critical for early horizons). |
| 7 | `item_id` | 138,136 | SKU identity encoding. |
| 8 | `rstd_7` | 109,265 | Short-term 7-day volatility. |
| 9 | `rmean_7` | 105,232 | Short-term 7-day level. |
| 10 | `dayofmonth` | 71,229 | Intramonth seasonality (payday and SNAP effects). |
| 11 | `dayofweek` | 69,878 | Day-of-week profile (weekend vs weekday shopping peaks). |
| 12 | `weekofyear` | 63,083 | Annual seasonality / holiday cycles. |
| 13 | `store_id` | 39,923 | Store location encoding. |
| 14 | `year` | 25,942 | Long-term store maturation trend. |
| 15 | `price_change_ratio` | 23,772 | Promotion / price discount signal. |

---

## 6. Execution and Reproducibility

To reproduce Milestone 9 training and evaluation from scratch:

```bash
# Run tests for Milestone 9
python -m pytest tests/test_lightgbm_forecaster.py -v

# Run full hyperparameter tuning and backtest
python -m scripts.run_lightgbm_forecaster

# Or run with default hyperparameters without tuning
python -m scripts.run_lightgbm_forecaster --no-tune
```

Artifacts generated:
- `data/processed/lightgbm/predictions.parquet`: 26,880 predictions
- `data/processed/lightgbm/scored_predictions.parquet`: Scored development forecasts
- `data/processed/lightgbm/metrics.csv`: Fold and pooled metrics
- `data/processed/lightgbm/feature_importance.csv`: Average feature gains
- `data/processed/lightgbm/comparison_metrics.csv`: Complete cross-model comparison table
- `reports/m9_lightgbm_comparison.csv`: Model comparison table
