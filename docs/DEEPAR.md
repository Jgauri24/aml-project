# DeepAR Global Probabilistic Forecasting Model: Technical Specification & Guide

This document specifies the architecture, mathematics, data pipeline, training dynamics, autoregressive inference, and integration interfaces of the **DeepAR Global Probabilistic Forecasting Model** implemented for the AML M5 Inventory Demand Forecasting project.

---

## 1. What DeepAR Is

DeepAR (Salinas et al., *International Journal of Forecasting*, 2020) is an autoregressive probabilistic recurrent neural network model designed for forecasting across large collections of heterogeneous time series. Unlike local, single-series models (e.g. ARIMA, exponential smoothing, or independent Kalman filters) that fit separate parameters for each SKU, DeepAR is a **global model**: a single shared neural network parameterized by recurrent weights $\Theta$ learns common seasonal patterns, calendar effects, and cross-series demand dynamics jointly across all time series in a dataset.

At each time step $t$, the recurrent network maintains a latent internal state $(h_t, c_t)$ and outputs the parameters of a conditional probability distribution over the next observation:

$$P(y_{i,t} \mid y_{i,1:t-1}, x_{i,1:t}) = p(y_{i,t} \mid \theta(h_{i,t}))$$

During future multi-step forecasting, DeepAR generates future sample paths **autoregressively**: samples drawn from the predictive distribution at step $t$ are fed back as the input to step $t+1$.

---

## 2. Why DeepAR for Retail Demand Forecasting

Retail demand at the SKU-store level presents several distinct characteristics:
1. **Intermittency and Sparsity**: Many SKU-store combinations have frequent zero-sales days (slow-moving products), while others have high-volume daily sales.
2. **Count Nature**: Demand is non-negative and discrete ($y_{i,t} \in \{0, 1, 2, \dots\}$). Continuous Gaussian approximations predict negative sales or distort zero-inflation.
3. **Cross-Series Regularities**: Items in the same category or department (e.g., `FOODS_1` in `CA_1`) share common calendar responses (weekend surges, holiday stock-ups) even if their baseline volume differs by an order of magnitude.
4. **Asymmetric Cost of Uncertainty**: In inventory optimization, stockout costs often exceed holding costs. Point forecasts cannot calculate safety stock or service levels; full predictive quantiles ($P10, P50, P90$) are required for probabilistic inventory policies.

DeepAR addresses all four challenges:
- Global parameter sharing transfers strength from high-volume series to intermittent series.
- Negative Binomial likelihood natively models overdispersed non-negative integer count demand.
- Dynamic scale normalization handles cross-series volume differences of up to 3 orders of magnitude.
- Monte Carlo autoregressive rollouts yield full predictive distributions and calibrated quantile prediction intervals.

---

## 3. Global vs. Local Forecasting

| Dimension | Local Forecasters (Kalman Filter, ARIMA) | Global Forecaster (DeepAR) |
|---|---|---|
| **Parameter Ownership** | Independent set of state/variance parameters per SKU-store. | Single shared parameter set $\Theta$ trained jointly over all series. |
| **Cross-Series Learning** | Zero information transfer across series. | Shared recurrent representations learn universal calendar and seasonal dynamics. |
| **Cold Start / Sparse SKUs** | Struggles with low volume or intermittent history. | Borrows statistical strength from related series via categorical embeddings. |
| **Model Size Scaling** | $\mathcal{O}(N)$ models to store and maintain for $N$ series. | $\mathcal{O}(1)$ neural network regardless of number of series. |
| **Covariate Ingestion** | Complex matrix expansions per series. | Native concatenation of static embeddings and dynamic covariates into input vector. |

---

## 4. Recurrent Network Architecture

The recurrent core is a multi-layer Long Short-Term Memory (LSTM) network.

### Latent State Dynamics
At sequence step $t$ for series $i$:
$$f_t = \sigma(W_f x_{i,t} + U_f h_{i,t-1} + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i x_{i,t} + U_i h_{i,t-1} + b_i) \quad \text{(input gate)}$$
$$\tilde{c}_t = \tanh(W_c x_{i,t} + U_c h_{i,t-1} + b_c) \quad \text{(candidate cell state)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell state update)}$$
$$o_t = \sigma(W_o x_{i,t} + U_o h_{i,t-1} + b_o) \quad \text{(output gate)}$$
$$h_t = o_t \odot \tanh(c_t) \quad \text{(hidden state)}$$

### Real-Time Mean Scaling
To normalize demand across series with vast volume differences, DeepAR computes a per-series scale factor $\nu_i$:
$$\nu_i = 1 + \frac{1}{|T_{\text{hist}}|} \sum_{t \in T_{\text{hist}}} y_{i,t}$$
The $+1$ offset ensures $\nu_i \ge 1$, avoiding division by zero for all-zero or sparse series.

### Autoregressive Input Vector
At time step $t$, the input $x_{i,t} \in \mathbb{R}^{D_{\text{in}}}$ is constructed as:
$$x_{i,t} = \begin{bmatrix} \log\left(1 + \frac{y_{i,t-1}}{\nu_i}\right) \\ e_i \\ z_{i,t} \end{bmatrix}$$
where:
- $\log(1 + y_{i,t-1} / \nu_i)$ is the log-transformed, normalized previous target.
- $e_i \in \mathbb{R}^{D_{\text{static}}}$ is the concatenated static categorical embedding vector.
- $z_{i,t} \in \mathbb{R}^{D_{\text{cov}}}$ is the vector of dynamic known calendar covariates at time $t$.

---

## 5. Negative Binomial Likelihood Formulation

Retail demand is count-valued and exhibits overdispersion ($\text{Var}(Y) > \mathbb{E}[Y]$). The Negative Binomial distribution $\text{NegBinomial}(\mu, \alpha)$ is parameterized by mean $\mu > 0$ and dispersion parameter $\alpha > 0$:

$$\mathbb{E}[Y] = \mu, \quad \text{Var}(Y) = \mu + \alpha \mu^2$$

As $\alpha \to 0$, the distribution converges to Poisson ($\text{Var} = \mu$). When $\alpha > 0$, overdispersion is captured quadratically with demand.

### Distribution Parameter Heads
The linear heads project the top LSTM hidden state $h_{i,t}$ to the distribution parameters:
$$\mu_{i,t} = \nu_i \cdot \text{softplus}(W_\mu h_{i,t} + b_\mu) + \epsilon$$
$$\alpha_{i,t} = \text{softplus}(W_\alpha h_{i,t} + b_\alpha) + 10^{-4}$$

### PyTorch Mapping & Numerical Stability
PyTorch parameterizes `NegativeBinomial(total_count=r, logits=logits)` where $r$ is the number of failures and $p$ is the probability of success:
$$r_{i,t} = \frac{1}{\alpha_{i,t}}$$
$$\text{logits}_{i,t} = \log\left(\frac{p}{1 - p}\right) = \log(\alpha_{i,t} \cdot \mu_{i,t})$$

The exact log-likelihood for observed count $y \ge 0$ is:
$$\log P(y \mid r, \text{logits}) = \ln\Gamma(y + r) - \ln\Gamma(r) - \ln\Gamma(y + 1) - r \log(1 + e^{\text{logits}}) + y \cdot \text{logits} - y \log(1 + e^{\text{logits}})$$

Training minimizes the negative log-likelihood (NLL):
$$\mathcal{L}(\Theta) = - \frac{1}{B \cdot T} \sum_{i=1}^B \sum_{t=1}^T \log P(y_{i,t} \mid \mu_{i,t}(\Theta), \alpha_{i,t}(\Theta))$$

---

## 6. Categorical Embeddings & Dynamic Covariates

### Static Categorical Embeddings
For each series, static metadata (`item_id`, `store_id`, `dept_id`, `cat_id`, `state_id`) is mapped via a fitted `StaticCategoryEncoder`. Index $0$ is explicitly reserved for `<unk>` (unknown categories) to prevent inference crashes on unseen entities.

| Feature | Vocab Size | Embedding Dimension |
|---|---|---|
| `item_id` | $N_{\text{items}} + 1$ | 16 |
| `store_id` | 11 | 4 |
| `dept_id` | 8 | 4 |
| `cat_id` | 4 | 2 |
| `state_id` | 4 | 2 |
| **Total $D_{\text{static}}$** | — | **28** |

### Known-Future Dynamic Covariates
Only features genuinely known in advance at forecast origin are included:
1. `wday_sin`, `wday_cos`: $\sin, \cos(2\pi (\text{wday} - 1) / 7)$ capturing weekly cycle.
2. `month_sin`, `month_cos`: $\sin, \cos(2\pi (\text{month} - 1) / 12)$ capturing annual seasonality.
3. `is_weekend`: Binary flag for Saturday and Sunday.
4. `has_event_1`, `has_event_2`: Binary flags indicating scheduled holiday / promotional events.
5. `day_progress`: Normalized continuous trend $(t - 1) / 1941.0$.

Future actual sales, future sell prices, and future SNAP indicators are strictly excluded per the conservative project protocol.

---

## 7. Autoregressive Inference & Quantile Generation

For origin $t_0$ and forecast horizon $H = 28$:

1. **Context Warmup**:
   The context window $y_{i, t_0 - T_{\text{ctx}} + 1 : t_0}$ is fed through the LSTM with teacher forcing to compute the terminal hidden state $(h_{i, t_0}, c_{i, t_0})$.

2. **Parallel Monte Carlo Rollout**:
   The recurrent state and static embeddings are expanded $S$ times ($S = 200$ sample trajectories per series):
   $$\text{Batch size for rollout} = N_{\text{series}} \times S = 240 \times 200 = 48,000 \text{ parallel paths}$$

3. **Sequential Autoregressive Step ($h = 1, \dots, 28$)**:
   - At step $h$, input is $x_{t_0 + h}^{(s)} = \left[ \log(1 + \hat{y}_{t_0 + h - 1}^{(s)} / \nu_i), e_i, z_{t_0 + h} \right]$.
   - Step the LSTM cell: $(h_h^{(s)}, c_h^{(s)}) = \text{LSTMCell}(x_{t_0+h}^{(s)}, (h_{h-1}^{(s)}, c_{h-1}^{(s)}))$.
   - Compute distribution parameters $(\mu_h^{(s)}, \alpha_h^{(s)})$.
   - Draw sample: $\hat{y}_{t_0 + h}^{(s)} \sim \text{NegBinomial}(\mu_h^{(s)}, \alpha_h^{(s)})$.
   - Enforce non-negativity: $\hat{y}_{t_0 + h}^{(s)} = \max(0, \hat{y}_{t_0 + h}^{(s)})$.
   - Pass $\hat{y}_{t_0 + h}^{(s)}$ as the feedback input to step $h + 1$.

4. **Quantile Extraction**:
   From the empirical distribution of the 200 trajectories:
   - $\text{P10} = \text{Quantile}(\hat{y}_{i, h}^{(1:S)}, 0.10)$
   - $\text{P50} = \text{Quantile}(\hat{y}_{i, h}^{(1:S)}, 0.50)$ (Median forecast, assigned to `y_pred`)
   - $\text{P90} = \text{Quantile}(\hat{y}_{i, h}^{(1:S)}, 0.90)$
   - Nonnegativity and ordering $0 \le P10 \le P50 \le P90$ are mathematically guaranteed.

---

## 8. Integration with AML Project Architecture

DeepAR implements the shared `BaseForecaster` interface defined in `src/evaluation/backtesting.py`:

```python
from src.evaluation.backtesting import BaseForecaster, ForecastInputs
from src.models.deepar_config import DeepARConfig
from src.models.deepar_forecaster import DeepARForecaster, make_deepar_factory

# Model instantiation and backtesting
deepar_cfg = DeepARConfig(context_length=56, hidden_dim=64, epochs=25)
factory = make_deepar_factory(deepar_cfg)
predictions, metrics, metadata = run_backtest(panel, cfg, factory, output_dir="deepar")
```

The output DataFrame strictly adheres to `PREDICTION_COLUMNS`:
`["model", "series_id", "origin", "horizon", "target_day_num", "target_date", "y_pred", "q10", "q50", "q90"]`

---

## 9. Hardware & Platform Compatibility

- **Apple Silicon (MPS)**: Fully supported and verified. Training loss minimization runs natively on MPS. For the Monte Carlo sampling loop, `NegativeBinomial.sample()` safely falls back to CPU to circumvent the missing `aten::_standard_gamma` MPS kernel in PyTorch.
- **CUDA**: Automatically selected if available.
- **CPU**: Default reliable fallback across all platforms.
- **No external heavy dependencies**: Eliminates fragile GluonTS/Lightning/HuggingFace library collisions on Python 3.13+. Uses standard PyTorch 2.x and NumPy/Pandas.

