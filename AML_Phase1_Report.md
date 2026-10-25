# Inventory Demand Forecasting via Local State-Space Kalman Filtering and Global Gradient Boosted Decision Trees

**Comprehensive Phase 1 Evaluation on the Walmart M5 Dataset**

Applied Machine Learning (CSAI) – Project 5 · Team 5
Department of Computer Science and Artificial Intelligence, School of Technology, Rishihood University, Sonipat, Haryana, India
Codebase: <https://github.com/Jgauri24/aml-project> · October 2026

| Team Member | Enrollment Number |
|---|---|
| Tanish Garg | 2401010473 |
| Ayush Mittal | 2401020091 |
| Gauri Jindal | 2401020022 |
| Sanchit Garg | 2401010414 |
| Saumya Soni | 2401020058 |

---

## Abstract

Accurate multi-horizon inventory demand forecasting is vital for retail supply-chain efficiency, working-capital optimisation, and stockout reduction. Retail sales series are difficult to model: they are highly intermittent (69.33% of daily records are zero), show a pronounced weekly cycle, and react to promotional shocks. This report compares two contrasting forecasting paradigms — local state-space tracking (five Kalman-filter variants) and a global, direct multi-horizon LightGBM model — against two seasonal heuristic baselines. Experiments use a deterministic, stratified cohort of 240 Walmart M5 series and an expanding four-fold rolling-origin backtest with a 28-day horizon (26,880 out-of-sample predictions); the final competition window is kept sealed.

Global LightGBM achieves the best point accuracy (MAE 0.9227, WAPE 0.7640). The Kalman local-level model with weekly seasonality achieves the lowest RMSE (2.0535) and, like the other Kalman variants, produces closed-form 80% prediction intervals (81.08% empirical coverage for the single-SKU model and 85–86% for the other variants). These results motivate a dual-model operating policy: LightGBM for procurement sizing and Kalman predictive variance for safety stock.

**Keywords:** demand forecasting, state-space models, Kalman filter, maximum likelihood estimation, gradient boosting, LightGBM, rolling-origin backtesting, uncertainty quantification, M5 competition.

## Contents

1. [Introduction](#1-introduction)
2. [Dataset and Cutoff-Safe Exploratory Analysis](#2-dataset-and-cutoff-safe-exploratory-analysis)
3. [State-Space Modelling and Kalman Filtering](#3-state-space-modelling-and-kalman-filtering)
4. [Feature Engineering and Global LightGBM](#4-feature-engineering-and-global-lightgbm)
5. [Experimental Protocol](#5-experimental-protocol)
6. [Results](#6-results)
7. [Discussion](#7-discussion)
8. [Interactive Diagnostic Platform](#8-interactive-diagnostic-platform)
9. [Conclusion and Operational Recommendations](#9-conclusion-and-operational-recommendations)
10. [References](#references)

---

## 1. Introduction

Modern supply-chain operations need forecasting systems that support two distinct decisions:

1. **Point demand sizing** — accurate conditional-mean or median estimates used to size procurement batches, warehouse logistics, and replenishment orders.
2. **Safety-stock sizing** — calibrated prediction intervals and variance bounds that cushion against demand volatility without inflating holding costs.

Retail daily sales, as in the Walmart M5 dataset [5], violate many textbook assumptions. They are intermittent (many zero-demand days), show asymmetric promotional spikes (for example, Supplemental Nutrition Assistance Program (SNAP) disbursement days), have stepped price changes, and follow a 7-day calendar cycle.

Practitioners typically choose between two families of methods:

- **Local time-series models** such as ARIMA, Croston's method and state-space Kalman filters [1], [3]. Parameters are fitted independently per series; they adapt quickly to level shifts and handle missing observations natively, but cannot share structure across sparse series.
- **Global tabular machine-learning models** such as gradient boosted decision trees (LightGBM [4]) and deep networks [2]. They pool histories from many items so that slow-moving products can borrow statistical strength from fast-moving ones.

### 1.1 Objectives and Contributions

1. A reproducible, stratified 240-series cohort and a cutoff-safe exploratory analysis (no look-ahead).
2. Five per-series Kalman state-space variants with maximum-likelihood hyper-parameter estimation and closed-form predictive quantiles.
3. A leakage-safe 26-feature pipeline and a direct multi-horizon LightGBM forecaster trained with an L1 loss.
4. A four-fold rolling-origin backtest (26,880 predictions) evaluated with point, error-ratio and probabilistic metrics.
5. An operational recommendation that combines the strengths of both paradigms.

---

## 2. Dataset and Cutoff-Safe Exploratory Analysis

### 2.1 Walmart M5 Dataset and Cohort Selection

The data source is the Walmart M5 hierarchical sales dataset: daily unit sales for items in stores across California (CA), Texas (TX) and Wisconsin (WI). To keep experiments reproducible and tractable, a deterministic, stratified cohort of $N=240$ series is built with a fixed seed (42) by `src/data/select_skus.py`:

- **Volume stratification:** exactly 80 low-, 80 medium- and 80 high-volume series, split by trailing-365-day sales up to the cutoff day `d1801`.
- **Hierarchical diversity:** the cohort spans all 3 states, all 10 stores, all 3 categories (FOODS, HOUSEHOLD, HOBBIES) and all 7 departments.
- **No artificial selection bias:** no minimum-positive-sales rule is applied, so genuinely intermittent series are retained.

### 2.2 Key Findings of the Exploratory Analysis

To avoid look-ahead bias, all exploratory analysis uses only days `d1`–`d1801` (2011-01-29 to 2016-01-03; 432,240 series-days).

| Aspect | Finding |
|---|---|
| Zero inflation | 69.33% of all daily records are exact zeros; 89.06% in the low-volume group (median CV 3.27; longest zero streak 1,640 days). |
| Demand pattern (Syntetos–Boylan) | 175 intermittent, 54 lumpy, 8 smooth, 3 erratic series. |
| Weekly seasonality | ACF peaks at lags 7, 14, 21, 28; median $r_7 = 0.1639$ (up to 0.3164 for high-volume SKUs); weekend-to-weekday peak-to-trough ratio 1.369×. |
| SNAP effect | State-specific lifts in the first 10 days of each month: CA +4.8%, TX +6.0%, WI +8.0%. |
| Price dynamics | Stepped, infrequent reductions; median price–demand correlation −0.0471; temporary rollbacks cause sharp spikes. |

Because the share of zero actuals is so high, percentage metrics such as MAPE are undefined or diverge when $y_t = 0$. We therefore evaluate with the Weighted Absolute Percentage Error (WAPE) and the Mean Absolute Error (MAE) instead.

---

## 3. State-Space Modelling and Kalman Filtering

State-space models treat observed daily sales $y_t$ as a noisy reflection of an unobserved demand state $\alpha_t$.

### 3.1 Linear Gaussian State-Space System

$$y_t = Z_t \alpha_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, R)$$

$$\alpha_t = T_t \alpha_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q)$$

where $Z_t$ is the observation vector, $T_t$ the transition matrix, $R > 0$ the observation variance and $Q \succeq 0$ the process-noise covariance.

### 3.2 Evaluated Kalman Variants

| Model | State | Description |
|---|---|---|
| `kalman_single_sku` | $[\mu_t]$ | Milestone-6 reference local-level model; $T=[1]$, $Z=[1]$. |
| `kalman_local_level` | $[\mu_t]$ | Random-walk level: $y_t = \mu_t + \varepsilon_t$, $\mu_t = \mu_{t-1} + \eta_t$. |
| `kalman_local_level_seasonal` | 7-dim | Level plus a 6-dimensional weekly seasonal vector $\gamma_t$ with $\sum_{j=0}^{6}\gamma_{t-j}=0$. |
| `kalman_local_trend` | $[\mu_t, \beta_t]$ | Level plus damped slope ($\rho = 0.98$), $T = \begin{pmatrix}1&1\\0&\rho\end{pmatrix}$, $Z=[1,0]$. |
| `kalman_local_trend_seasonal` | 8-dim | Damped trend combined with weekly seasonality. |

For the seasonal variants the state is $\alpha_t = [\mu_t, \gamma_{t,1}, \dots, \gamma_{t,6}]^\top$ with $Z = [1,1,0,0,0,0,0]$. The $6\times 6$ seasonal transition block has ones on the superdiagonal and a final row of $-1$'s, which enforces the zero-sum constraint above.

### 3.3 Recursive Filtering and Missing Data

Let $a_{t|t-1} = E[\alpha_t \mid \mathcal{F}_{t-1}]$ and $P_{t|t-1} = \mathrm{Var}(\alpha_t \mid \mathcal{F}_{t-1})$. With a diffuse prior $P_0 = 10^4 I$:

**Time update (predict):**

$$a_{t|t-1} = T\,a_{t-1|t-1}, \qquad P_{t|t-1} = T\,P_{t-1|t-1}\,T^\top + Q$$

**Measurement update (correct), when $y_t$ is observed:**

$$v_t = y_t - Z a_{t|t-1} \quad \text{(innovation)}$$

$$S_t = Z P_{t|t-1} Z^\top + R \quad \text{(innovation variance)}$$

$$K_t = P_{t|t-1} Z^\top S_t^{-1} \quad \text{(Kalman gain)}$$

$$a_{t|t} = a_{t|t-1} + K_t v_t, \qquad P_{t|t} = (I - K_t Z) P_{t|t-1}$$

**Missing-data invariant.** If $y_t$ is unobserved (`NaN`), the update is skipped: $a_{t|t} = a_{t|t-1}$ and $P_{t|t} = P_{t|t-1}$. The filter coasts on its transition model rather than fabricating zeros.

### 3.4 Parameter Estimation by Maximum Likelihood

The hyper-parameters $\theta = (\ln Q, \ln R)$ are estimated per series by minimising the exact prediction-error negative log-likelihood

$$\ell(\theta) = \frac{1}{2}\sum_{t=1}^{T}\left[\ln(2\pi S_t) + \frac{v_t^2}{S_t}\right]$$

using L-BFGS-B with multiple restarts in log-variance space ($\theta \in [-10, 8]$) to reduce the risk of poor local minima.

### 3.5 Multi-Step Predictive Distribution

For horizon $h \in \{1,\dots,H\}$ from origin $T$ (local-level case):

$$E[y_{T+h} \mid \mathcal{F}_T] = \mu_{T|T}, \qquad V_h = \mathrm{Var}(y_{T+h} \mid \mathcal{F}_T) = P_{T|T} + hQ + R$$

so predictive variance grows linearly with the horizon. Quantiles are clipped at zero because sales cannot be negative:

$$\hat q_\tau(y_{T+h}) = \max\!\left(0,\ \hat y_{T+h} + z_\tau \sqrt{V_h}\right), \qquad z_{0.10} = -1.282,\ z_{0.90} = +1.282$$

---

## 4. Feature Engineering and Global LightGBM

### 4.1 Cross-Series Learning

Whereas each Kalman filter learns its own parameters, the global model pools all 240 series into one tabular training set. This lets low-velocity products borrow promotional, calendar and seasonal effects learned from dense items.

### 4.2 Leakage-Safe Feature Pipeline (Milestone 8)

`src/features/feature_engineering.py` builds 26 features anchored strictly at the forecast origin $T$ (only information with $t \le T$):

- **Demand lags:** sales at $T-1$, $T-7$, $T-14$, $T-28$.
- **Rolling statistics:** 7-day and 28-day rolling means and standard deviations over $[T-W, T]$.
- **Calendar encodings:** day-of-week, day-of-month, month, weekend flag and circular sine/cosine transforms.
- **Exogenous indicators:** state SNAP flags (`snap_CA`, `snap_TX`, `snap_WI`), national event flags and sell price.
- **Direct-horizon invariant:** for horizon $h$, any lag feature with lag $< h$ is set to `NaN` and handled by LightGBM's native missing-value branching. A dedicated check function (`assert_no_future_leakage()`) enforces this.

### 4.3 LightGBM Forecaster (Milestone 9)

- **Loss:** L1 (`regression_l1`), so the trees target the conditional median, which is robust to zero inflation and holiday spikes.
- **Tuning:** rolling-validation tuning over the first three backtest folds selected a learning rate of 0.05, 500 estimators, 31 leaves per tree and a feature fraction of 0.8.

---

## 5. Experimental Protocol

### 5.1 Four-Fold Chronological Backtest

We use an expanding-window rolling-origin backtest with horizon $H = 28$ days. Each fold contains $240 \times 28 = 6{,}720$ forecasts, giving 26,880 out-of-sample predictions in total. The final competition period (`d1914`–`d1941`) is kept as a sealed holdout and is never used.

| Fold | Origin | Test window | Forecasts |
|---|---|---|---|
| 1 | `d1801` | 2016-01-04 to 2016-01-31 | 6,720 |
| 2 | `d1829` | 2016-02-01 to 2016-02-28 | 6,720 |
| 3 | `d1857` | 2016-02-29 to 2016-03-27 | 6,720 |
| 4 | `d1885` | 2016-03-28 to 2016-04-24 | 6,720 |

### 5.2 Evaluation Metrics

With $M$ evaluated points:

$$\mathrm{MAE} = \frac{1}{M}\sum_i |y_i - \hat y_i|, \qquad \mathrm{RMSE} = \sqrt{\frac{1}{M}\sum_i (y_i - \hat y_i)^2}$$

$$\mathrm{WAPE} = \frac{\sum_i |y_i - \hat y_i|}{\sum_i y_i}, \qquad \mathrm{PL}_\tau = \max\big(\tau(y - \hat q_\tau),\ (1-\tau)(\hat q_\tau - y)\big)$$

The empirical coverage $\mathrm{PICP}_{80}$ is the share of actuals falling in $[\hat q_{0.10}, \hat q_{0.90}]$ (nominal 80%). Pinball loss and coverage are reported only for the Kalman models, which produce predictive quantiles.

---

## 6. Results

### 6.1 Pooled Point-Forecast Accuracy

**Table 1.** Point-forecast accuracy across all folds (26,880 series-days). Best value per column in bold.

| Model | Paradigm | MAE ↓ | RMSE ↓ | WAPE ↓ |
|---|---|---|---|---|
| LightGBM (global) | Global tabular GBDT | **0.9227** | 2.1081 | **0.7640** |
| Kalman LL Seasonal | Local state-space | 1.0007 | **2.0535** | 0.8286 |
| Kalman LT Seasonal | Local state-space | 1.0030 | 2.0576 | 0.8305 |
| Kalman Local Level | Local state-space | 1.0106 | 2.0613 | 0.8368 |
| Kalman Single-SKU (M6) | Local state-space | 1.0115 | 2.0776 | 0.8375 |
| Kalman Local Trend | Local state-space | 1.0225 | 2.0976 | 0.8466 |
| Naive Weekly Seasonal | Heuristic baseline | 1.1869 | 2.5487 | 0.9828 |
| Naive Last Value | Heuristic baseline | 1.2479 | 2.5151 | 1.0332 |

![Figure 1: MAE, RMSE and WAPE for all eight models](figures/metrics_overview.png)

*Figure 1. MAE, RMSE and WAPE for all eight models (LL = local level, LT = local trend). LightGBM (green) leads on MAE and WAPE; the seasonal local-level Kalman model leads on RMSE.*

**Main observations**

- **Point accuracy:** LightGBM reduces MAE by 7.8% relative to the best Kalman model (1.0007) and by 22.3% relative to the best naive baseline, weekly seasonal (1.1869); against the last-value baseline the reduction is 26.1%.
- **Squared error:** the seasonal local-level Kalman model has the lowest RMSE (2.0535 vs. 2.1081 for LightGBM). RMSE penalises large errors heavily, so this suggests the Kalman forecasts make fewer large misses on high-demand days, although the pooled results alone do not prove the cause.
- **Trend states do not help:** the local-trend model has the worst Kalman WAPE (0.8466). Retail demand oscillates around a level rather than drifting steadily, so a trend component adds variance without benefit.

### 6.2 Comparison of the Kalman Variants

The five Kalman variants are close in pooled WAPE (0.8286–0.8466): adding weekly seasonality helps slightly (local level 0.8368 → 0.8286), whereas adding a damped trend hurts (0.8466). The differences are small compared with the gap to LightGBM or to the naive baselines.

![Figure 2: Development WAPE of the five Kalman variants](figures/kalman_wape.png)

*Figure 2. Development WAPE of the five Kalman variants. The bars are nearly equal in height; the seasonal local-level model is marginally lowest.*

### 6.3 Probabilistic Forecasts: Pinball Loss and Coverage

Only the Kalman models produce predictive intervals. Every variant covers at least the nominal 80%. The single-SKU model is closest to nominal (81.08%), whereas the other four variants over-cover at roughly 85–86%, i.e. their intervals are somewhat conservative (wider than necessary). For safety-stock purposes, over-coverage errs on the side of protection but implies extra holding cost.

**Table 2.** Probabilistic metrics for the Kalman models. Best value per column in bold.

| Model | PL Q10 ↓ | PL Q50 ↓ | PL Q90 ↓ | 80% coverage |
|---|---|---|---|---|
| Kalman LL Seasonal | 0.1318 | **0.5004** | 0.3306 | 85.72% |
| Kalman LT Seasonal | 0.1347 | 0.5015 | **0.3304** | 85.43% |
| Kalman Local Level | **0.1315** | 0.5053 | 0.3366 | 85.64% |
| Kalman Single-SKU (M6) | 0.1426 | 0.5058 | 0.3518 | **81.08%** |
| Kalman Local Trend | 0.1419 | 0.5112 | 0.3371 | 85.37% |

![Figure 3a: Empirical coverage](figures/coverage.png)

*Figure 3a. Empirical coverage of the nominal 80% interval.*

![Figure 3b: Pinball loss](figures/pinball.png)

*Figure 3b. Pinball loss at the 10th, 50th and 90th percentiles.*

### 6.4 Performance by Volume Band

The table below reports the best Kalman model (local level + seasonal) by volume band. The cohort is balanced (8,960 observations per band), yet relative error rises sharply as volume falls: WAPE is 0.7166 for high-volume series but exceeds 1.0 for medium (1.1136) and low volume (1.2740), because the denominator $\sum y_i$ is tiny when most days are zero. MAE moves the opposite way (1.9453 → 0.3566) because absolute demand is lower.

**Table 3.** Kalman local level + seasonal by volume band.

| Volume band | Observations | MAE ↓ | RMSE ↓ | WAPE ↓ |
|---|---|---|---|---|
| High | 8,960 | 1.9453 | 3.3110 | 0.7166 |
| Medium | 8,960 | 0.7002 | 1.1062 | 1.1136 |
| Low | 8,960 | 0.3566 | 0.6813 | 1.2740 |

![Figure 4: WAPE and MAE by volume band](figures/band_metrics.png)

*Figure 4. Kalman local level + seasonal: WAPE (left; dashed line marks WAPE = 1) and MAE (right) by volume band.*

### 6.5 Fold-by-Fold Stability

The ranking is the same in every fold: LightGBM is best, followed by Kalman LL seasonal, then the naive weekly seasonal and naive last-value baselines. Every model records its highest MAE in the final fold (origin `d1885`), although LightGBM stays below 1.0 throughout (0.8821–0.9679) and the last-value baseline barely changes after fold 1 (1.2314–1.2564). No single fold drives the conclusions.

**Table 4.** MAE across the four rolling folds. Best per fold in bold.

| Model | Fold 1801 | Fold 1829 | Fold 1857 | Fold 1885 |
|---|---|---|---|---|
| LightGBM | **0.8821** | **0.9306** | **0.9103** | **0.9679** |
| Kalman LL Seasonal | 0.9723 | 1.0145 | 0.9903 | 1.0257 |
| Naive Weekly Seasonal | 1.1460 | 1.1835 | 1.1676 | 1.2507 |
| Naive Last Value | 1.2314 | 1.2528 | 1.2509 | 1.2564 |

![Figure 5: MAE by fold](figures/mae_by_fold.png)

*Figure 5. MAE by forecast origin. LightGBM is best at every origin; all models find the last fold hardest.*

### 6.6 Naive Baselines Across Origins

The pooled WAPE of the two heuristics on the development origins is shown below. The weekly seasonal baseline is at or below the last-value baseline at every origin (roughly 0.97–1.00 vs. 0.99–1.07), confirming that the 7-day cycle carries real signal. Both hover around a WAPE of 1.0, meaning a naive forecast is barely better than predicting nothing at all — the bar that the learned models must clear.

![Figure 6: Naive baseline WAPE by origin](figures/naive_by_origin.png)

*Figure 6. Pooled WAPE of the naive baselines by development origin.*

### 6.7 Qualitative Forecast Examples

We inspect one representative series per volume band. The figures below show the naive forecasts at origin `d1801`. For the high-volume series `FOODS_1_004__TX_2` the last-value baseline stays flat at 1 unit while actual sales climb in mid-January with spikes up to 13 units, so both baselines miss the peaks by a wide margin; the weekly seasonal baseline reproduces a repeating pattern that does not match the realised timing. For the low-volume series `FOODS_1_038__WI_2` sales are sporadic 0/1 events, and flat or periodic forecasts cannot anticipate which days will be non-zero. The medium-volume example `FOODS_1_026__CA_2` is all zeros in this window, so every baseline is trivially correct there — a reminder of how heavily zeros dominate this cohort.

![Figure 7a: Naive baselines, high volume](figures/cmp_high.png)

*Figure 7a. High volume (`FOODS_1_004__TX_2`).*

![Figure 7b: Naive baselines, low volume](figures/cmp_low.png)

*Figure 7b. Low volume (`FOODS_1_038__WI_2`).*

![Figure 7c: Naive baselines, medium volume](figures/cmp_medium.png)

*Figure 7c. Medium volume (`FOODS_1_026__CA_2`).*

The Kalman variants with their $q_{10}$–$q_{90}$ bands at the last origin (`d1885`) show three behaviours:

- **Median forecasts hug zero.** For intermittent series the $q_{50}$ forecast sits near zero; the seasonal variants add a small weekly ripple (e.g. peaks around 0.4 units for the low-volume series) while the non-seasonal variants are flat.
- **Bands widen with the horizon,** consistent with $V_h = P_{T|T} + hQ + R$, and reach about 1 unit for the low- and medium-volume series.
- **Wide upper bands after long zero runs.** For the high-volume series the realised sales are zero throughout the window, yet the upper band rises to roughly 3–5 units. The model is uncertain because the series was active earlier in the year; here the interval is conservative, which fits the observed over-coverage in Table 2. For the medium-volume series, there is no sales history in the plotted training window, and the series then records occasional sales of 1–3 units which the median forecast cannot anticipate, although most fall inside the 80% band.

![Figure 8a: Kalman variants, high volume](figures/kv_high.png)

*Figure 8a. High volume (`FOODS_1_004__TX_2`), origin `d1885`.*

![Figure 8b: Kalman variants, low volume](figures/kv_low.png)

*Figure 8b. Low volume (`FOODS_1_038__WI_2`), origin `d1885`.*

![Figure 8c: Kalman variants, medium volume](figures/kv_medium.png)

*Figure 8c. Medium volume (`FOODS_1_026__CA_2`), origin `d1885`.*

---

## 7. Discussion

### 7.1 Why LightGBM Wins on Point Accuracy

Global pooling lets the model learn calendar, SNAP, price and lag effects from all 240 series at once, whereas each Kalman filter must infer its parameters from a single, mostly-zero series. The L1 objective also targets the median, which suits zero-heavy data and is consistent with the MAE and WAPE metrics on which LightGBM leads. The working hypothesis is that local maximum-likelihood estimation of $(Q, R)$ becomes unreliable when about 89% of days are zero (low-volume band); a per-band breakdown for LightGBM would be needed to confirm this directly.

### 7.2 Why Kalman Filters Remain Valuable

- **Calibrated uncertainty for free:** predictive variance $V_h$ is analytic, and all five variants reach at least the nominal 80% coverage.
- **Interpretability:** the latent level, trend and seasonal states have direct business meaning.
- **Missing-data handling:** the filter skips updates for unobserved days instead of imputing values.
- **Best RMSE:** the seasonal local-level model gives the lowest squared error of all models.

### 7.3 Limitations

- **Gaussian predictive intervals** are only an approximation for non-negative, count-like, zero-heavy demand; zero-clipping fixes the sign but not the shape of the distribution.
- **Over-coverage** (85–86%) in four of the five variants shows the intervals are not perfectly calibrated and could be tightened.
- **No probabilistic output for LightGBM** in this phase, so the two paradigms cannot yet be compared on pinball loss or coverage. A quantile-objective LightGBM would address this.
- **Cohort size:** 240 series and four folds are a small sample of the full M5 hierarchy, and pooled WAPE for low- and medium-volume series exceeds 1.0 for the Kalman model reported.
- **Exogenous variables:** the Kalman variants here use no SNAP, price or event regressors, which LightGBM does.

---

## 8. Interactive Diagnostic Platform

To bridge theory and practice we built a Streamlit application (`app.py`) with three components:

1. **Live Kalman simulation:** sliders for $Q$ and $R$ show the effect on the Kalman gain $K_t$ and on the widening 80% prediction intervals.
2. **Leakage-safe feature explorer:** real-time construction of the 26 tabular features at a chosen forecast origin $T$.
3. **Interactive leaderboards:** bar charts and metric sorting across all eight models.

---

## 9. Conclusion and Operational Recommendations

Neither local state-space models nor a global tree-based forecaster is a complete solution on its own. Our results support a hybrid policy:

1. **Procurement baseline:** use global LightGBM for purchase-order sizing, since it has the lowest expected unit error (WAPE 0.7640, MAE 0.9227).
2. **Safety-stock sizing:** use the Kalman predictive variance $V_h$ to set dynamic, calibrated safety-stock buffers (81.08% empirical coverage for the single-SKU model at a nominal 80%).

**Phase 2 roadmap.** Future work will combine the two modalities in global probabilistic recurrent architectures such as DeepAR [2] and Temporal Fusion Transformers, using Kalman-filtered states as dynamic structural inputs, and will add quantile-based LightGBM so that all models can be compared on probabilistic metrics.

---

## References

1. R. E. Kalman, "A new approach to linear filtering and prediction problems," *Journal of Basic Engineering*, vol. 82, no. 1, pp. 35–45, 1960.
2. D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, "DeepAR: Probabilistic forecasting with autoregressive recurrent networks," *International Journal of Forecasting*, vol. 36, no. 3, pp. 1181–1191, 2020.
3. A. C. Harvey, *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press, 1990.
4. G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
5. S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "The M5 accuracy competition: Results, findings and conclusions," *International Journal of Forecasting*, vol. 38, no. 4, pp. 1346–1364, 2022.
6. J. E. Boylan and A. A. Syntetos, *Intermittent Demand Forecasting: Context, Methods and Applications*. John Wiley & Sons, 2021.
