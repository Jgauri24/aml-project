# Inventory Demand Forecasting via Kalman Filtering and Global LightGBM Models
**Phase 1 Machine Learning Technical Report — Applied Machine Learning (CSAI)**

**Authors:** Team 5 (CSAI Project Group)  
**Target Dataset:** M5 Forecasting - Accuracy (Walmart Hierarchical Unit Sales)  
**Evaluation Track:** Phase 1 ML Component (Literature, EDA, Feature Engineering, State-Space Models, GBDT, & Backtesting)  
**Repository:** `https://github.com/Jgauri24/aml-project`  

---

## Executive Summary & Abstract

Accurate inventory demand forecasting is foundational to supply-chain efficiency, directly driving stock availability, working-capital optimization, and waste minimization. Retail sales data, however, present severe modeling challenges: high intermittent sparsity (zero-demand days), asymmetric holiday demand shocks, price elasticity shifts, and calendar-induced weekly seasonality. 

In this work, we design, implement, and benchmark an end-to-end Machine Learning demand forecasting system comparing **local state-space models** against **global sequence gradient-boosted decision trees (LightGBM)** across a deterministic cohort of 240 representative M5 retail series over four rolling-origin backtest folds ($28$-day forecast horizon, $26,880$ total evaluated series-days). 

Our primary methodological contributions include:
1. **Local State-Space Tracking (Kalman Filtering - Milestone 6 & 7):** Formulated single-SKU and multi-SKU local-level Gaussian state-space models. Latent demand level and noise parameters $(Q, R)$ are estimated per-series via Maximum Likelihood Estimation (MLE) using the prediction-error innovations decomposition. We deliver closed-form predictive distributions that provide well-calibrated $80\%$ prediction intervals ($81.08\%$ empirical coverage vs. $80.00\%$ nominal target).
2. **Leakage-Safe Feature Engineering (Milestone 8):** Engineered 26 direct-horizon tabular features—including demand lags, rolling aggregates, calendar dynamics, and SNAP promotion indicators—under strict temporal non-peeking constraints verified by automated invariant suites (`assert_no_future_leakage`).
3. **Global Tabular Forecaster (LightGBM - Milestone 9):** Trained a unified gradient boosted model across pooled series histories, optimizing an L1 median objective that achieves the lowest pooled Mean Absolute Error (**$0.9227$**) and Weighted Absolute Percentage Error (**$0.7640$**), outperforming naive heuristics by over $26\%$.
4. **Interactive Verification & Diagnostic UI:** Implemented a full-stack Streamlit interactive dashboard enabling live Kalman parameter sensitivity tuning, real-time feature synthesis, and exhaustive metric visualisations.

---

## 1. Problem Definition & Literature Review

### 1.1 Problem Formulation
Let $y_{i,t} \in \mathbb{R}_{\ge 0}$ denote the unit sales of SKU-store combination $i \in \{1, \dots, N\}$ on discrete day $t \in \{1, \dots, T\}$. For a forecast origin $T$, our objective is to predict future demand over a fixed horizon $H = 28$ days:
$$\hat{y}_{i, T+h} \quad \text{for } h \in \{1, \dots, H\}$$
For probabilistic models, the objective is to generate conditional quantile trajectories $q_\tau(y_{i, T+h} \mid \mathcal{F}_T)$ for nominal coverage levels $\tau \in \{0.10, 0.50, 0.90\}$, where $\mathcal{F}_T$ represents all observable information up to time $T$.

### 1.2 Theoretical Foundation & Related Work
* **Classical State-Space Modeling (Kalman, 1960; Harvey, 1990):** The linear Gaussian state-space model posits that observed demand $y_t$ reflects an unobserved underlying level $\mu_t$ contaminated by observation noise. Kalman filters offer recursive, exact analytical updates with zero lookahead bias and handle missing observations natively without artificial imputation.
* **Intermittent Demand Forecasting (Croston, 1972; Syntetos & Boylan, 2005):** Traditional retail sales exhibit high zero proportions. While Croston-type methods separate non-zero demand size from inter-arrival times, state-space formulations capture level drift smoothly while accumulating predictive variance during periods of unobserved demand or stockouts.
* **Global vs. Local Forecasting (Januschowski et al., 2020; Montero-Manso & Hyndman, 2021):** Local models fit parameters independently per time series. While adaptive, they cannot share cross-series statistical strength. Global models (such as LightGBM or DeepAR) fit a single unified architecture across all pooled series, enabling sparse or intermittent items to borrow seasonal and promotional dynamics from dense series.

---

## 2. Dataset Understanding & Exploratory Data Analysis (EDA)

### 2.1 The M5 Walmart Dataset Cohort
The M5 dataset comprises hierarchical daily unit sales across Walmart retail stores in California (CA), Texas (TX), and Wisconsin (WI). To ensure tractable, reproducible, and mathematically rigorous backtesting without loss of generality, we constructed a deterministic, stratified cohort of **240 SKU-store series**:
* Stratified across **3 states** (CA, TX, WI) and **10 stores**.
* Segmented across all **3 categories** (`FOODS`, `HOUSEHOLD`, `HOBBIES`) and **7 departments**.
* Balanced across **volume bands** (High, Medium, Low volume) and intermittency levels.

### 2.2 Key EDA Findings
1. **Pervasive Intermittency & Zero Inflation:** Zero-demand days account for a significant fraction of observations ($38.2\%$ in low-volume series), necessitating models that avoid divergence when computing relative percentage errors.
2. **Strong Day-of-Week Seasonality:** Autocorrelation functions (ACF) reveal significant spikes at lag 7 and lag 14 ($r_7 \in [0.08, 0.32]$). Weekend sales (Saturday/Sunday) exhibit a $25\%\text{--}40\%$ volume elevation over mid-week days.
3. **Calendar Events and SNAP Elasticity:** Supplemental Nutrition Assistance Program (SNAP) purchase allowances produce distinct, state-specific demand surges during the first 10–15 days of each month.
4. **Price Elasticity & Promotions:** Price changes are stepped and infrequent; temporary price rollbacks correspond to demand spikes.

---

## 3. Methodology: Local State-Space Models (Kalman Filter)

### 3.1 Mathematical Formulation of the Local-Level Model (Milestone 6)
The local-level state-space model decomposes daily sales into a dynamic latent level $\mu_t$ and independent Gaussian noise:

$$\begin{aligned}
\text{Observation Equation:} \quad & y_t = \mu_t + \varepsilon_t, \quad &\varepsilon_t \sim \mathcal{N}(0, R) \\
\text{State Equation:} \quad &\mu_t = \mu_{t-1} + \eta_t, \quad &\eta_t \sim \mathcal{N}(0, Q)
\end{aligned}$$

* **$Q \ge 0$ (Process Noise Variance):** Dictates how rapidly the latent demand level drifts over time.
* **$R > 0$ (Observation Noise Variance):** Measures daily idiosyncratic sales volatility around the level.

### 3.2 Recursive Forward Filtering
Given initial prior $\mu_0$ and diffuse prior variance $P_0 = 10^4$, the recursive filter executes two steps at each time step $t$:
1. **Time Update (Predict):**
   $$\mu_{t \mid t-1} = \mu_{t-1 \mid t-1}, \qquad P_{t \mid t-1} = P_{t-1 \mid t-1} + Q$$
2. **Measurement Update (Correct):**
   When $y_t$ is observed:
   $$v_t = y_t - \mu_{t \mid t-1} \quad (\text{Innovation}), \qquad S_t = P_{t \mid t-1} + R \quad (\text{Innovation Variance})$$
   $$K_t = \frac{P_{t \mid t-1}}{S_t} \quad (\text{Kalman Gain})$$
   $$\mu_{t \mid t} = \mu_{t \mid t-1} + K_t v_t, \qquad P_{t \mid t} = (1 - K_t) P_{t \mid t-1}$$
   *Handling Missing Data / Stockouts:* If $y_t$ is unobserved or marked as a stockout (`NaN`), the observation update is bypassed:
   $$\mu_{t \mid t} = \mu_{t \mid t-1}, \qquad P_{t \mid t} = P_{t \mid t-1}$$

### 3.3 Maximum Likelihood Estimation via Prediction-Error Decomposition
The parameters $\theta = (\ln Q, \ln R)$ are estimated per-series by minimizing the exact negative log-likelihood:
$$\ell(\theta) = \frac{1}{2} \sum_{t=1}^T \left[ \ln(2\pi S_t) + \frac{v_t^2}{S_t} \right]$$
We employ L-BFGS-B with multiple deterministic restarts covering distinct parameter regimes (neutral $Q=R=1$, slow-drift $Q \ll R$, and responsive $R \ll Q$).

### 3.4 Multi-Step Forecasting & Predictive Distribution
Because the state follows a random walk, the conditional point forecast is invariant across the horizon:
$$\mathbb{E}[y_{T+h} \mid \mathcal{F}_T] = \mu_{T \mid T} \quad \forall h \in \{1, \dots, H\}$$
Predictive variance accumulates linearly with forecast distance $h$:
$$V_h = \text{Var}(y_{T+h} \mid \mathcal{F}_T) = P_{T \mid T} + h \cdot Q + R$$
The predictive density is Gaussian: $y_{T+h} \mid \mathcal{F}_T \sim \mathcal{N}(\mu_{T \mid T}, V_h)$. Non-negativity is enforced via zero-clipping:
$$\hat{y}_{T+h} = \max(0, \mu_{T \mid T}), \qquad q_\tau(y_{T+h}) = \max\left(0, \mu_{T \mid T} + z_\tau \sqrt{V_h}\right)$$

### 3.5 Multi-SKU Kalman Variants (Milestone 7)
In Milestone 7, we extended the baseline filter into four structured variants:
1. `kalman_local_level`: Standard random walk level.
2. `kalman_local_level_seasonal`: Augments the state vector with a 6-dimensional seasonal state to model the 7-day cyclical weekly pattern.
3. `kalman_local_trend`: Incorporates a dynamic slope component $\beta_t$ ($\mu_t = \mu_{t-1} + \beta_{t-1} + \eta_t$).
4. `kalman_local_trend_seasonal`: Full structural model combining level, slope, and weekly seasonal harmonics.

---

## 4. Methodology: Feature Engineering & Global LightGBM

### 4.1 Leakage-Safe Feature Engineering Pipeline (Milestone 8)
Global models require rich tabular feature representations. To guarantee zero temporal lookahead bias, our `FeatureEngineer` constructs 26 features at each forecast origin $T$:
* **Demand Lags:** Direct lags at $t - 1, t - 7, t - 14, t - 28$.
* **Rolling Statistics:** 7-day and 28-day rolling means and standard deviations computed exclusively over $t \le T$.
* **Calendar Dynamics:** Integer day-of-week, day-of-month, month, weekend indicators, and cyclical sine/cosine transforms.
* **SNAP & Event Flags:** Binary indicators for state-specific SNAP qualification days and major national calendar events.
* **Direct-Horizon Invariant:** For prediction steps $h > \text{lag}$, future values are treated strictly as missing (`NaN`), branching through LightGBM's native missingness pathways.

### 4.2 LightGBM Forecaster Architecture (Milestone 9)
* **Objective:** L1 regression loss (`regression_l1` / MAE), aligning model predictions with the conditional median.
* **Hyperparameter Selection:** Tuned via chronological rolling validation on the first three development folds. Selected configuration: `num_leaves=31`, `learning_rate=0.05`, `n_estimators=500`.

---

## 5. Experimental Protocol & Evaluation Setup

### 5.1 Rolling-Origin Backtesting Architecture (Milestone 3 & 4)
To ensure empirical realism, we implement a rolling-origin evaluation over **four development folds**, each with an evaluation horizon of $H = 28$ days:

| Fold Index | Forecast Origin ($T$) | Evaluation Calendar Period | Test Observations |
| :---: | :---: | :---: | :---: |
| Fold 1 | Day $1801$ | 2016-01-04 to 2016-01-31 | 6,720 |
| Fold 2 | Day $1829$ | 2016-02-01 to 2016-02-28 | 6,720 |
| Fold 3 | Day $1857$ | 2016-02-29 to 2016-03-27 | 6,720 |
| Fold 4 | Day $1885$ | 2016-03-28 to 2016-04-24 | 6,720 |
| **Total** | — | **Full Development Span** | **26,880** |

*Final Holdout Reservation:* The final competition window (Days 1914–1941) was strictly isolated and unaccessed during training and tuning.

### 5.2 Quantitative Evaluation Metrics
1. **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{M} \sum_{i,h} |y_{i, T+h} - \hat{y}_{i, T+h}|$
2. **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{M} \sum_{i,h} (y_{i, T+h} - \hat{y}_{i, T+h})^2}$
3. **Weighted Absolute Percentage Error (WAPE):** $\text{WAPE} = \frac{\sum_{i,h} |y_{i, T+h} - \hat{y}_{i, T+h}|}{\sum_{i,h} y_{i, T+h}}$
4. **Quantile Pinball Loss ($\text{PL}_\tau$):**
   $$\text{PL}_\tau(y, \hat{q}_\tau) = \max\left(\tau(y - \hat{q}_\tau), (1 - \tau)(\hat{q}_\tau - y)\right)$$
5. **Empirical Prediction Interval Coverage (PICP):** Proportion of true observations falling within $[\hat{q}_{0.10}, \hat{q}_{0.90}]$ (Nominal target: $80\%$).

---

## 6. Experimental Results & Comparative Analysis

### 6.1 Pooled Benchmark Results Across All Models

The table below compiles the definitive benchmark results evaluated across all 240 series and four backtest folds ($26,880$ evaluated observations):

| Model Architecture | Model Family | Pooled MAE ↓ | Pooled RMSE ↓ | Pooled WAPE ↓ | Pinball Q10 ↓ | Pinball Q50 ↓ | Pinball Q90 ↓ | $80\%$ Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Global)** | Global GBDT | **0.9227** | 2.1081 | **0.7640** | — | — | — | — |
| `kalman_local_level_seasonal` | State-Space | 1.0007 | **2.0535** | 0.8286 | 0.1318 | 0.5004 | 0.3306 | 85.72% |
| `kalman_local_trend_seasonal` | State-Space | 1.0030 | 2.0576 | 0.8305 | 0.1347 | 0.5015 | 0.3304 | 85.43% |
| `kalman_local_level` | State-Space | 1.0106 | 2.0613 | 0.8368 | 0.1315 | 0.5053 | 0.3366 | 85.64% |
| **`kalman_single_sku` (M6)** | State-Space | 1.0115 | 2.0776 | 0.8375 | 0.1426 | 0.5058 | 0.3518 | **81.08%** |
| `kalman_local_trend` | State-Space | 1.0225 | 2.0976 | 0.8466 | 0.1419 | 0.5112 | 0.3371 | 85.37% |
| `naive_weekly_seasonal` | Heuristic | 1.1869 | 2.5487 | 0.9828 | — | — | — | — |
| `naive_last_value` | Heuristic | 1.2479 | 2.5151 | 1.0332 | — | — | — | — |

### 6.2 Key Scientific Insights

1. **Global Sharing Superiority on L1 Error:** LightGBM achieves the best pooled MAE ($0.9227$) and WAPE ($0.7640$), representing a **$22.3\%$ reduction in MAE** over the naive seasonal baseline. By pooling 240 series, the model successfully transfers seasonal and promotional elasticities to sparse series.
2. **Harmonic State-Space Superiority on Squared Error:** The `kalman_local_level_seasonal` model achieves the best overall pooled RMSE ($2.0535$), outperforming even LightGBM ($2.1081$). By explicitly modeling weekly harmonic state vectors, the seasonal Kalman filter tracks peak weekend volumes without the high-variance errors produced by tree splits.
3. **Calibrated Uncertainty Quantification:** `kalman_single_sku` delivers empirical coverage of **$81.08\%$**, closely tracking the nominal $80\%$ target. This provides warehouse managers with rigorous confidence bounds for safety-stock determination.
4. **Trend Model Degeneracy:** Adding an unconstrained linear trend (`kalman_local_trend`) degraded MAE ($1.0225$) and RMSE ($2.0976$). In retail sales, demand does not grow monotonically; local trend components overfit temporary replenishment surges.

---

## 7. Interactive System Architecture (Streamlit UI)

To bridge theoretical rigor and operational decision-making, we engineered an interactive web application ([`app.py`](../app.py)):

```
┌────────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT WEB APPLICATION (app.py)                   │
├───────────────────┬────────────────────────────────────────────────────┤
│ 🏠 Overview       │ Milestone map (M0–M9), architecture, data stats    │
│ 📊 Comparison     │ Dynamic metric sorting, color-coded leaderboard    │
│ 🔮 Live Kalman    │ Live MLE vs. Manual (Q, R) sliders, 80% intervals  │
│ ⚡ Live LightGBM  │ Feature vector synthesizer, multi-step projection  │
│ 🔍 Diagnostics    │ Volume-band breakdown, empirical coverage tables   │
│ 📈 EDA Explorer   │ ACF, seasonality, sparsity, and SNAP distributions │
└───────────────────┴────────────────────────────────────────────────────┘
```

* **Interactive Kalman Experimentation:** Allows examiners to manually modulate $Q$ and $R$, demonstrating in real time how the steady-state Kalman gain $K_t$ shifts trust between prior belief and noisy sales.
* **Leakage Verification:** Displays real-time feature vectors generated at origin $T$ to verify zero lookahead bias.

---

## 8. Viva Voce & Individual Assessment Guide

### Key Theoretical Questions & Defensible Answers

**Q1: Why is the multi-step point forecast for the local-level Kalman filter constant?**  
*Answer:* The state transition is formulated as a pure martingale (random walk): $\mu_t = \mu_{t-1} + \eta_t$ with $\mathbb{E}[\eta_t] = 0$. Consequently, the conditional expectation of future levels given origin information $\mathcal{F}_T$ is invariant: $\mathbb{E}[\mu_{T+h} \mid \mathcal{F}_T] = \mu_{T \mid T}$ for all $h \ge 1$. While the point prediction is flat, uncertainty expands with horizon: $V_h = P_{T \mid T} + hQ + R$.

**Q2: What is the physical interpretation of the Kalman Gain $K_t$?**  
*Answer:* $K_t = \frac{P_{t \mid t-1}}{P_{t \mid t-1} + R} \in [0, 1]$ represents the optimal weight assigned to the latest innovation $v_t = y_t - \mu_{t \mid t-1}$. When observation noise $R$ is large relative to prior variance, $K_t \to 0$, discounting the daily reading as noise. When process noise $Q$ is high, prior variance $P_{t \mid t-1}$ grows, causing $K_t \to 1$ and rapidly shifting the level toward the newest observation.

**Q3: How does the Kalman Filter handle stockouts and missing sales?**  
*Answer:* If sales are unobserved (`NaN`), the measurement update step is bypassed. The posterior level remains equal to the prior prediction ($\mu_{t \mid t} = \mu_{t \mid t-1}$), while the state covariance grows ($P_{t \mid t} = P_{t \mid t-1} + Q$), naturally expanding uncertainty until demand is observed again.

**Q4: How did you ensure your feature pipeline has zero future data leakage?**  
*Answer:* In Milestone 8, all lag and rolling features are anchored strictly at origin day $T$. For any horizon step $h > \text{lag}$, future values are treated as missing (`NaN`). This contract is validated by automated regression tests (`tests/test_feature_engineering.py`) and invariant assertions (`assert_no_future_leakage`).

---

## 9. Conclusion & Phase 2 Outlook

Phase 1 successfully establishes a rigorous machine learning benchmark comparing local state-space filters and global gradient boosting across 240 M5 series. The empirical results demonstrate that while global tree models maximize point accuracy through cross-series feature pooling, local state-space models excel in seasonal squared-error stability and calibrated uncertainty quantification.

**Phase 2 Roadmap (Deep Learning & Hybrid Innovation):**
1. **Global Probabilistic Sequence Models:** Implement DeepAR (autoregressive recurrent neural networks with Gaussian/Negative Binomial likelihoods) and Temporal Fusion Transformers (TFT).
2. **Technically Justified Hybrid Architecture:** Feed Kalman filtered state estimates and innovation residuals as dynamic features into the global deep sequence model, combining local state tracking with global sequence learning.
