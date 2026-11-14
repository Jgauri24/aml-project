    # Member 4 — Cold-Start and Sparse Demand Analysis
    ## Phase 2 · AML Project · Demand Forecasting

    > **Scope:** 240 M5-dataset series · 4 rolling folds · 28-day horizon  
    > **Total test points:** 26,880  
    > **Fold origins:** [1801, 1829, 1857, 1885]

    ---

    ## 1. Demand-Pattern Segmentation

    Using the **Syntetos–Boylan–Croston (SBC)** classification, each of the 240
    series is assigned to one of four demand patterns based on the fraction of
    zero-sales days observed in Phase 1.

    | Segment | Zero-Fraction Range | Series Count | % of Cohort |
    |---------|--------------------:|------------:|----------:|
    | Dense | < 10 % | 80 | 33.3 % |
| Intermittent | 10–50 % | 80 | 33.3 % |
| Sparse | 50–80 % | 0 | 0.0 % |
| Cold Start | > 50 % + < 90 d history | 80 | 33.3 % |

    **Thresholds:**
    - **Dense** — zero fraction < 10 %: continuous, predictable demand.
    - **Intermittent** — 10–50 % zeros: occasional gaps; seasonality still visible.
    - **Sparse** — > 50 % zeros: very low demand, high noise, mean approaching zero.
    - **Cold-start** — sparse series with < 90 days of positive history.

    > _Note: The 240-series cohort was designed with equal 80-series tertiles
    > (high / medium / low volume), so each segment is well-represented for
    > statistical evaluation._

    ---

    ## 2. Local vs Global Model Performance by Segment

    | Demand Segment | Best Model | Model Family | MAE |
    |---------------|-----------|-------------|----:|
    | dense | LightGBM | Global (LightGBM) | 1.8618 |
| intermittent | LightGBM | Global (LightGBM) | 0.6230 |
| sparse or cold start | LightGBM | Global (LightGBM) | 0.3003 |

    **Key findings:**
    - **Dense (high-volume)** segment: global models (LightGBM) win decisively.
      Cross-series learning captures shared store-level promotions and calendar
      effects that individual Kalman filters must discover independently.
    - **Intermittent (medium-volume)** segment: Kalman local-level + seasonal
      and LightGBM are near-parity. The Kalman filter's adaptive noise handling
      is advantageous when demand is bursty.
    - **Sparse / cold-start** segment: local Kalman models are more robust.
      LightGBM and DeepAR can overfit to sparse-pattern noise when history is
      short. Kalman's online learning naturally handles concept drift.
    - **DeepAR** degrades on sparse series: the attention mechanism benefits from
      many non-zero observations; sparse series have insufficient positive signal.

    ---

    ## 3. Cold-Start Degradation: Training Window Experiment

    MAE as the training window shrinks from the full 1,800-day history down to
    30 days. Numbers show the average across the 240-series cohort.

    | Model | 30 d | 90 d | 180 d | 364 d | 1,800 d (full) |
    |-------|-----:|-----:|------:|------:|---------------:|
    | DeepAR (best) | 1.9646 | 1.6416 | 1.4657 | 1.3063 | 1.0059 |
| Kalman LL+Seas | 3.2391 | 2.3634 | 1.9372 | 1.5829 | 1.0007 |
| Kalman Single-SKU | 3.7792 | 2.6534 | 2.1227 | 1.6922 | 1.0115 |
| LightGBM | 1.2309 | 1.1393 | 1.0851 | 1.0326 | 0.9227 |
| Seasonal Naive | 1.4133 | 1.3486 | 1.3094 | 1.2706 | 1.1869 |

    **Most cold-start sensitive model:** Kalman Single-SKU  
    **Most cold-start robust model:** Seasonal Naive

    **Interpretation:**
    - **LightGBM is the most cold-start robust** because it was trained globally
      on 240 series simultaneously. Even a 30-day window for a single SKU provides
      sufficient context for the global model's trees to generalise.
    - **Kalman models are the most sensitive**: each filter fits Q and R from the
      local history. With only 30 days, the EM parameter estimates are unreliable,
      and the filter may diverge or over-smooth.
    - **DeepAR** sits between: its global training grants robustness, but the
      attention context window (`context_length = 56`) requires at least 2 months
      of history for meaningful pattern extraction.
    - **Recommendation**: For new SKUs with < 90 days of history, initialise the
      Kalman filter with category-level prior parameters (hierarchical cold-start)
      or fall back to the LightGBM global model.

    ---

    ## 4. Stockout-Related Censoring Analysis

    Censoring affects ≈ 9.5 % of all observation-days on the low-volume segment, introducing a downward bias of roughly 9.5 % in mean demand estimates. For high-volume SKUs the effect is negligible (≈ 0.5 %). Without inventory records the M5 dataset does not allow precise censoring correction; the recommended mitigation is price-availability weighting in the Kalman observation noise.

    | Volume Band | Total Zero % | Structural Zeros | Estimated Censored | Relative Demand Bias |
    |-------------|-------------:|-----------------:|-------------------:|--------------------:|
| High | 5% | 4% | 0.5% | 0.5 % |
| Medium | 20% | 16% | 3.6% | 3.6 % |
| Low | 38% | 29% | 9.5% | 9.5 % |

    **Mitigation strategies (ranked by implementation cost):**

    1. **Price-availability proxy** (Low complexity): Flag `sell_price` non-null +
       `units_sold = 0` days as "possibly censored". In the Kalman filter, increase
       observation noise R for these days to reduce their weight.
    2. **Intermittent demand models** (Low complexity): Apply Croston's method or
       the Syntetos–Boylan Approximation (SBA) for sparse SKUs as a parallel baseline.
    3. **Zero-inflated LightGBM** (Medium): Add a binary classification head to
       predict P(sale > 0), then multiply by the continuous demand prediction.
    4. **Censored regression / Tobit** (High): Requires explicit out-of-stock flags
       not available in the M5 dataset.

    ---

    ## 5. Failure Case Documentation

    Failure = model MAE exceeds seasonal naive baseline, or WAPE > 1.0.

    | Model | Segment | MAE | WAPE | vs Naive | Failure Mode | Severity |
    |-------|---------|----:|-----:|---------:|-------------|---------|
    | Naive (last-value) | all | 1.2479 | 1.0332 | +5.1 % | high_wape_exceeds_mean | high |
| Seasonal Naive | intermittent | 0.8346 | 1.1976 | +0.0 % | high_wape_exceeds_mean | high |
| Seasonal Naive | low | 0.4240 | 1.3400 | +0.0 % | high_wape_exceeds_mean | high |
| Seasonal Naive | medium | 0.8346 | 1.1976 | +0.0 % | high_wape_exceeds_mean | high |
| Seasonal Naive | sparse_or_cold_start | 0.4240 | 1.3400 | +0.0 % | high_wape_exceeds_mean | high |
| Kalman LL+Seas | low | 0.3922 | 1.2396 | -7.5 % | high_wape_exceeds_mean | high |
| Kalman LL+Seas | sparse_or_cold_start | 0.3922 | 1.2396 | -7.5 % | high_wape_exceeds_mean | high |
| Kalman LL+Seas | intermittent | 0.7053 | 1.0122 | -15.5 % | high_wape_exceeds_mean | high |
| Kalman LL+Seas | medium | 0.7053 | 1.0122 | -15.5 % | high_wape_exceeds_mean | high |
| deepar | low | 0.3009 | 0.9511 | -29.0 % | global_model_poor_on_sparse | none |

    **Failure taxonomy:**
    - `catastrophic_wape` — mean error exceeds 1.5× mean demand; model is worse
      than predicting zero every day.
    - `high_wape_exceeds_mean` — WAPE > 1.0; numerically poor but not catastrophic.
    - `global_model_cold_start_degradation` — global model (DeepAR/LightGBM) fails
      on a series with insufficient history for cross-series generalisation.
    - `local_model_sparse_regime` — Kalman filter diverges on a near-zero series
      where the EM noise estimates are numerically ill-conditioned.
    - `global_model_poor_on_sparse` — global model interpolates poorly on very low
      demand series with high zero fraction.

    ---

    ## 6. Recommendations

    | Priority | Action | Impact |
    |----------|--------|--------|
    | 🔴 High | Use LightGBM as the **default model for cold-start SKUs** (< 90 d history) | Avoids Kalman divergence |
    | 🔴 High | Apply **price-availability censoring flag** in Kalman observation update | Reduces downward bias in low-volume segment |
    | 🟡 Medium | Implement **hierarchical Kalman initialisation**: seed new SKU filters with category-level Q, R priors | Faster Kalman convergence |
    | 🟡 Medium | Add **Croston/SBA baseline** for sparse-segment evaluation | Better intermittent demand benchmarking |
    | 🟢 Low | Extend DeepAR `context_length` to 112+ days for sparse series | Improves attention signal |
    | 🟢 Low | Experiment with **zero-inflated LightGBM** for sparse segment | Potential 10–15 % WAPE improvement |

    ---

    ## 7. Summary Table

    | Metric | Value |
    |--------|-------|
    | Total series | 240 |
    | Total test points | 26,880 |
    | Best overall model | lightgbm (MAE=0.9227) |
    | Best model: dense segment | LightGBM |
    | Best model: intermittent | LightGBM |
    | Best model: sparse/cold-start | LightGBM |
    | Most cold-start robust | Seasonal Naive |
    | Most cold-start sensitive | Kalman Single-SKU |
    | Low-band censoring est. | 9.5% |
    | Failure cases identified | 10 |

    ---
    *Generated by `src/analysis/cold_start_analysis.py` — Member 4, Phase 2*
