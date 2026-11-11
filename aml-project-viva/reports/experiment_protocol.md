# Experiment protocol: Milestones 0–4

## Objective and unit of analysis

Prepare a fair comparison of weekly seasonal naive, per-series Kalman, global LightGBM and global probabilistic DeepAR. None of those models is implemented or trained in these milestones. One series is one item/store pair, identified as `item_id + '__' + store_id`. The response is **observed daily unit sales**; latent unconstrained demand and inventory availability are not observed.

`experiment.yaml` is the source of truth. Dependencies are pinned in `requirements.txt`; actual artifact versions and configuration/source hashes are recorded. Data artifacts under `data/` and the virtual environment are ignored by Git. Figures, findings, notebooks and implementation are reviewable project outputs.

## Data boundary and source integrity

The existing four M5 CSVs are used; nothing is downloaded. `sales_train_evaluation.csv` alone supplies the canonical observed sales. The validation file overlaps exactly through d_1913 (30,490 series × 1,913 days = 58,327,370 target cells); its suffix-dependent `id` is excluded from the comparison. Both sales keys, calendar `d`/date, and weekly price composite keys are checked for uniqueness. Sales sources are never concatenated.

Raw shape/schema inspection and mechanical integrity checks are allowed over the file structure. Target previews, missingness inspection in the understanding notebook, selection statistics, representative choice and EDA stop at d_1801. The explicit source-integrity comparison through d_1913 is not an EDA, selection or tuning input. The full selected panel stores d_1–d_1941 to avoid a later alternate preprocessing path; final targets are only mechanically validated (keys, continuity, domain and missingness), never analyzed or evaluated. The development CLI uses Parquet filtering through d_1913, and the engine again removes later rows before building any fold.

## Deterministic cohort

- History used for all eligibility and selection: d_1–d_1801, inclusive.
- Primary ranking: trailing 365-day mean, d_1437–d_1801. Missing values are not invented.
- Eligibility: configured minimum observed fraction, default 1.0; all 30,490 source series are eligible in this data. No sales-volume, launch, or positive-sales requirement exists.
- Rank ascending by trailing mean and then `series_id`. Split ranks into three near-equal terciles. Mean ties may cross group boundaries and are resolved deterministically; reported numeric intervals can therefore overlap at endpoints.
- Seed 42, sample 80 per band. Round-robin over `(store_id, cat_id)` strata with seeded within-stratum and stratum-order randomness; a depleted stratum cannot contribute further. The selected cohort has 240 series, with all 10 stores and all 3 categories represented in each band. Eligibility, counts, band limits and compromises are recorded in `selection_metadata.json`.
- Balance is a pragmatic diversity heuristic, not proportional or exactly equal allocation. Sparse/all-zero candidates remain eligible; no positive sales is not proof that an item has launched. The sample is not representative of population product/store proportions.
- Selection IDs/statistics are saved to `selected_skus.csv`. Preprocessing recomputes selection from safe raw history; loading artifacts rejects changed config or selected-file hashes.

## Canonical preprocessing and quality

Filter selected series before melting the daily columns. Produce one daily row per `series_id`, preserve every indexed timestamp, then join calendar by `d` with many-to-one validation and require every day to match. Join weekly prices by `(item_id, store_id, wm_yr_wk)` with many-to-one validation; missing prices do not remove sales rows. Preserve the full price/event/calendar provenance.

The canonical panel is 240 × 1,941 = **465,840 rows**, from 2011-01-29 through 2016-05-22. Missing indexed rows are distinct from rows whose target is missing and from zeros on observed rows. Targets are never filled or forward-filled. Identifiers/events are categorical, day/year/week integers use small integer types, and numeric targets/prices use float32. M5 unit counts fit the exact-integer range of float32; metrics accumulate in float64. Parquet can change a categorical dictionary's string NA backend without changing values or the logical category dtype.

Flags have precise meanings:

| Flag | Meaning |
| --- | --- |
| `is_zero_sale` | True only for an observed target equal to zero; nullable if target is missing. |
| `price_missing` | No numeric weekly selling price joined for that daily row. |
| `possible_prelaunch` | No positive sale **and** no price have been observed from day 1 through the row timestamp. A deliberately uncertain past-only heuristic. |
| `availability_unknown` | Always true because no stock/inventory measurements exist. |

No suspected or confirmed stockout label is manufactured. No first-sale date calculated over all future history is used. An eventual launch is not backdated to earlier days. Historical flags are recomputed inside each fold from that fold's history, and none is supplied as a future covariate. These flags are audit/context fields, not validated inventory labels.

Validation fails on duplicate/null keys, missing/excess indexed rows, discontinuous dates/day numbers, inconsistent day/date mappings, changing static metadata, negative/infinite targets, nonpositive/infinite observed prices, inconsistent zero/price flags or unobserved availability presented as known. Missing target values are preserved and explicitly counted.

## Chronological development and final reservation

All training windows expand from d_1. Horizon is 28; development target windows do not overlap.

| Origin | Training | Target days | Target dates |
| --- | --- | --- | --- |
| 1801 | d_1–d_1801 | d_1802–d_1829 | 2016-01-04–2016-01-31 |
| 1829 | d_1–d_1829 | d_1830–d_1857 | 2016-02-01–2016-02-28 |
| 1857 | d_1–d_1857 | d_1858–d_1885 | 2016-02-29–2016-03-27 |
| 1885 | d_1–d_1885 | d_1886–d_1913 | 2016-03-28–2016-04-24 |
| **Reserved** 1913 | d_1–d_1913 | **d_1914–d_1941** | **2016-04-25–2016-05-22** |

The final row is documentation only. There is no final-evaluation switch in the development CLI. It must remain unused until later modeling, tuning and protocol choices are frozen. Sales-file validation/evaluation suffixes do not change this project's split definitions.

## Feature availability and model interface

`ForecastInputs` deliberately contains three separate tables: history, static metadata and known future calendar. It never contains a validation/holdout target object or the full original panel. `fit(inputs)` must fit all learned preprocessing and model state only on the historical table. `predict(inputs)` returns the requested future grid. No learned preprocessing exists yet.

| Information | Policy |
| --- | --- |
| Sales through origin | Allowed in history; missing values stay missing. |
| Item/store/category/department/state | Allowed, derived from historical static metadata. |
| Future dates, day/week/month/year, weekday, known calendar events | Allowed under an announced-calendar assumption. |
| Historical actual selling prices | Allowed in history. |
| Future actual selling prices | Prohibited; absent from future table. No future price imputation or oracle-price scenario is implemented. |
| Historical SNAP | Allowed in history, including each series state's `snap` flag. |
| Future SNAP | Conservatively excluded until a realistic known-in-advance schedule assumption is specified. |
| Future sales or target-derived rolling statistics | Prohibited; excluded by an explicit column allowlist. Recursive features later may use model predictions, never true future sales. |
| Future inventory/availability/quality flags | Unknown, excluded. |

Historical calendar/quality fields are projected explicitly; arbitrary columns added to the panel do not automatically become features. Future dates/horizons must match the origin's calendar. Later model implementations must not load raw targets, full panels or globally fitted scalers themselves; the engine cannot police arbitrary model code outside the supplied interface.

A model factory returns a **fresh** `BaseForecaster` per fold; reusing an instance is an error. No state transfers between folds. Models must implement their own missing-history policy inside fit if needed; framework preprocessing does not fill targets. Any global scaler, categorical encoder or feature selection that learns from targets/history must be fitted in that fold, with no prefit on the full panel.

## Predictions and evaluation

Standard forecast fields are `model, series_id, origin, horizon, target_day_num, y_pred, q10, q50, q90`, plus `target_date` for explicit alignment. Horizon is the offset 1–28. Output must cover every requested series/day exactly once and identify one model consistently across folds. Points must be finite and nonnegative. Either all quantiles are unavailable (NA) or all are finite, nonnegative and ordered `q10 <= q50 <= q90`; point forecasts need not equal the median. No values are silently clipped or quantiles rearranged.

Actuals are joined only inside the evaluator after forecast validation. Missing actuals are excluded from scoring; missing target keys cause a hard error. A finite forecast is required even on requested rows whose actual value may be missing. The original predictions are stored separately from scored predictions.

Metrics are **pooled across nonmissing observed series-days**, both within each fold and across all four folds; they are not averages of per-series metrics. Large series can dominate pooled errors, so group/series diagnostics are an appropriate later addition. We do not claim to reproduce the competition's hierarchical WRMSSE score.

- MAE: mean absolute error in units.
- RMSE: square root of mean squared error, in units.
- WAPE: sum absolute error / sum absolute actual sales; reported as a ratio, not percent. Zero denominator (including all-zero perfect predictions) is **NA**, never forced to zero or infinity.
- Pinball loss: mean `max(q * (y - prediction), (q - 1) * (y - prediction))` separately at q=0.1/0.5/0.9. At q=0.5 this is half the corresponding MAE. Point-only models report NA pinball; quantiles are never invented from point values.
- Empty/all-missing evaluation subsets report NA. Infinite actuals or missing predictions for observed actuals are rejected.

Metric implementation is checked against hand calculations and scikit-learn's [mean pinball loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_pinball_loss.html). Join checks use pandas' [merge cardinality validation](https://pandas.pydata.org/docs/reference/api/pandas.merge.html).

## Infrastructure result and safeguards

The zero-valued DummyForecaster exercises fit/predict/evaluate only; it is not the seasonal naive baseline or an experimental model result. It outputs **26,880 × 10** forecasts (240 × 4 × 28), with 6,720 per origin. All target days are <=1913. Pooled sanity checks: MAE approximately **1.207775**, RMSE **2.846187**, WAPE **1.0**, and unavailable quantile scores NA. Execution metadata captures train/prediction counts, fit/predict times, environment, config and source-panel hash.

Tests cover raw schemas and keys, deterministic selection and future-target mutation, sparse eligibility, joins, missing/zero preservation, past-only heuristics, continuity and static metadata, Parquet round trips, stale artifacts, exact splits, hand-calculated/library metrics, point/quantile validation, fresh models, evaluator-only actual joins, explicit feature allowlists, fold-local fitting, and final-holdout mutations leaving all development outputs unchanged. EDA outputs and model inputs also remain unchanged when future targets and future prices are mutated.

See `pytest_results.xml` for the actual test run and `implementation_handoff.md` for executed commands and final counts. The final holdout has not been scored. No seasonal naive, Kalman, LightGBM or DeepAR model exists yet.
