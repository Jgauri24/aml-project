# M7 independent Kalman findings

## Executive summary

Selection rule, stated before reading the comparison table: choose the lowest
pooled development WAPE; only an exact tie within `1e-12` is broken by lower
q50 pinball and then lower combined q10/q90 pinball. By that rule,
`kalman_local_level_seasonal` wins with WAPE 0.828557. The result is based only
on the four development origins and is not a final-holdout claim.

## Verified M6 baseline

The verified `kalman_single_sku` development run has 26,880 predictions and
26,880 observed targets, maximum target day 1913, and
`final_holdout_evaluated=false`. Its pooled MAE/RMSE/WAPE are
1.011527/2.077604/0.837512. Its q10/q50/q90 pinball losses are
0.142601/0.505763/0.351790.

## Model definitions

- `kalman_single_sku` (M6): independent local-level filter.
- `kalman_local_level`: M7 generic local-level state space, independently
  estimated per series.
- `kalman_local_level_seasonal`: local level plus six-state weekly dummy
  seasonality.
- `kalman_local_trend`: local level plus damped slope; damping is fixed at
  0.98 and was not tuned.
- `kalman_local_trend_seasonal`: damped trend plus weekly dummy seasonality.

All M7 variants estimate their own parameters from their own fit-window
history. They are not pooled or hierarchical.

## Data and development protocol

The cohort has 240 item/store series. Development origins are d_1801, d_1829,
d_1857 and d_1885, each with a 28-day target period, totaling 26,880 rows per
model through d_1913. The final holdout d_1914–d_1941 was not used or scored.
M7 uses a trailing 364-calendar-day fit window by default; this is a recency
and computational choice, not evidence that earlier history is irrelevant.

## Comparison table

Copied from `reports/kalman_variant_comparison.csv`.

| model | MAE | RMSE | WAPE | pinball q10 | pinball q50 | pinball q90 | coverage | mean width | fit seconds | predict seconds | fallbacks | nonconverged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kalman_single_sku | 1.011527 | 2.077604 | 0.837512 | 0.142601 | 0.505763 | 0.351790 | 0.810789 | 2.720044 | 185.954016 | 0.871693 | unavailable | unavailable |
| kalman_local_level | 1.010637 | 2.061347 | 0.836776 | 0.131462 | 0.505319 | 0.336644 | 0.856362 | 2.793631 | 58.110396 | 1.223637 | 1 | 1 |
| kalman_local_level_seasonal | 1.000711 | 2.053497 | 0.828557 | 0.131828 | 0.500356 | 0.330628 | 0.857180 | 2.762593 | 209.475019 | 1.215572 | 1 | 1 |
| kalman_local_trend | 1.022487 | 2.097607 | 0.846587 | 0.141941 | 0.511244 | 0.337118 | 0.853683 | 2.798389 | 170.225579 | 1.224391 | 1 | 1 |
| kalman_local_trend_seasonal | 1.003014 | 2.057588 | 0.830464 | 0.134743 | 0.501507 | 0.330363 | 0.854315 | 2.764325 | 281.147637 | 1.281408 | 1 | 1 |

M6 fallback/convergence diagnostics are unavailable because its historical
run does not contain an M7-style fit summary.

## Breakdowns

### Winner by volume band

By WAPE, `kalman_local_level_seasonal` wins the high band (0.716600),
`kalman_single_sku` wins the low band (1.264666), and
`kalman_local_trend_seasonal` wins the medium band (1.112746). Each model-band
cell has 8,960 observed rows and no missing actuals.

### Winner by horizon bucket

By WAPE, `kalman_local_trend_seasonal` wins days 1–7 (0.798790).
`kalman_local_level_seasonal` wins days 8–14 (0.832725), 15–21 (0.829598),
and 22–28 (0.853415). Every model-bucket cell has 6,720 observed rows and no
missing actuals.

### Results by origin

By WAPE, `kalman_local_trend_seasonal` is lowest at d_1801 (0.818645) and
d_1829 (0.840816). `kalman_local_level_seasonal` is lowest at d_1857
(0.848154) and d_1885 (0.805518). Each model-origin cell has 6,720 observed
rows and no missing actuals.

## Calibration and interval width

The nominal q10–q90 coverage is 0.80. Empirical coverage is 0.810789 for M6
and 0.856362, 0.857180, 0.853683, and 0.854315 for the four M7 variants in
comparison-table order. Mean interval widths are 2.720044, 2.793631,
2.762593, 2.798389, and 2.764325. These are descriptive development results,
not calibration guarantees.

Gaussian predictive distributions approximate count data. Negative Gaussian
points and quantiles are clipped at zero; this changes the distribution and
can change nominal coverage, especially for sparse series. The empirical
coverage above 0.80 for M7 is evidence of conservative intervals on these
development targets, not proof of correct calibration elsewhere.

## Runtime, convergence, and fallbacks

The saved per-fold fit/predict totals are in the comparison table. A total
command wall-clock duration was not persisted in the generated artifacts, so
it is not reported here. Each M7 model has 960 series fits: 959 converged and
one is a documented fallback.

The sole M7 fallback is `FOODS_1_102__CA_4` at origin d_1885, with reason
`all_zero_window`. No optimizer-failure fallback was recorded. The all-zero
fallback supplies finite nonnegative forecasts and ordered intervals; it is a
safeguard rather than a model-performance success.

## Seasonal and trend evidence

Seasonal variants beat `kalman_local_level` on pooled WAPE:
`kalman_local_level_seasonal` is 0.828557 versus 0.836776 and
`kalman_local_trend_seasonal` is 0.830464. The M7 synthetic weekly gate also
passed before the real run. This is evidence that weekly seasonality helped
under the fixed development protocol.

Trend does not help the selection criterion: `kalman_local_trend` has WAPE
0.846587 versus 0.836776 for local level, and trend-seasonal has 0.830464
versus 0.828557 for level-seasonal. The more complex trend variants are not
preferred simply for having more state components.

## Limitations and recommendations

Evidence: all comparison rows use the same series, origins, targets, metric
definitions, and development-only boundary. Speculation: the observed volume-
band differences may reflect sparsity and heterogeneous demand patterns, but
the aggregate tables do not establish a causal explanation.

Limits include Gaussian likelihood for integer counts, zero clipping, fixed
damping 0.98, the 364-day fit window, and independent per-series estimation.
Independence prevents cross-series borrowing, which can be inefficient for
sparse histories. Missing observations skip the Kalman measurement update and
still propagate state uncertainty; no imputation is performed. Sparse and
all-zero histories remain challenging, as the recorded fallback demonstrates.

No numerical crash occurred in the completed M7 run, but per-series MLE cost
is material and rises with state dimension. The next milestone should implement
a compatible weekly seasonal-naive baseline, compare it under this same
protocol, and reserve final-holdout evaluation until selection is frozen.
