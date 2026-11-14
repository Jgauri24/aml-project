"""Development-only calibration diagnostics from genuine row-level forecasts.

No fitting, interval fabrication, stockout inference, or independent-row
significance tests. Group labels use training history only.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.backtesting import (
    FUTURE_KEYS, PREDICTION_COLUMNS, attach_actuals, validate_predictions,
)
from src.evaluation.metrics import score_predictions
from src.evaluation.splits import development_splits


def history_labels(panel, cfg):
    """Fold-local volume terciles and trailing zero runs, before evaluation.

    Equal-sized volume groups break ties by series ID. They are descriptive
    ranks within the supplied cohort, not learned global thresholds.
    """
    records = []
    for fold in development_splits(cfg):
        history = panel.loc[panel.day_num.between(fold.train_start, fold.origin)]
        for series_id, frame in history.groupby('series_id', observed=True, sort=True):
            y = frame.sort_values('day_num').set_index('day_num').units_sold.reindex(
                range(fold.train_start, fold.origin + 1))
            # Missing observations break a zero run rather than counting as zero.
            zeros = y.eq(0).to_numpy()
            run = 0
            for value in zeros[::-1]:
                if not value:
                    break
                run += 1
            records.append(dict(series_id=str(series_id), origin=fold.origin,
                                history_mean=y.mean(), history_observed=int(y.notna().sum()),
                                history_zero_fraction=y.eq(0).sum() / y.notna().sum()
                                if y.notna().any() else np.nan,
                                trailing_zero_days=run))
    labels = pd.DataFrame(records)
    labels['volume_group'] = 'unobserved'
    for _, frame in labels.groupby('origin', sort=True):
        ordered = frame.loc[frame.history_mean.notna()].sort_values(['history_mean', 'series_id'])
        bins = np.minimum(np.arange(len(ordered)) * 3 // max(len(ordered), 1), 2)
        labels.loc[ordered.index, 'volume_group'] = np.array(['low', 'medium', 'high'])[bins]
    return labels


def prepare_scored(predictions, panel, cfg):
    """Require the full same grid for every model; reattach canonical actuals.

    The caller supplies the selected cohort's panel. Saved actual columns are
    deliberately ignored. Extra/absent origins, holdout rows, partial cohorts,
    duplicate keys and invalid quantiles are errors rather than silent filters.
    """
    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f'Missing forecast columns: {sorted(missing)}')
    forecasts = predictions[PREDICTION_COLUMNS].copy()
    forecasts['series_id'] = forecasts.series_id.astype(str)
    forecasts['target_date'] = pd.to_datetime(forecasts.target_date)
    if forecasts.empty or forecasts.model.isna().any():
        raise ValueError('Supply nonempty forecasts with model identifiers')
    panel = panel.copy()
    panel['series_id'] = panel.series_id.astype(str)
    panel['date'] = pd.to_datetime(panel.date)
    if panel.duplicated(['series_id', 'day_num']).any():
        raise ValueError('Duplicate canonical panel keys')
    grid = []
    series_ids = sorted(panel.series_id.unique())
    for fold in development_splits(cfg):
        expected = panel.loc[panel.day_num.isin(fold.target_days),
                             ['series_id', 'day_num', 'date']].rename(
                                 columns={'day_num': 'target_day_num', 'date': 'target_date'})
        if len(expected) != len(series_ids) * fold.horizon:
            raise ValueError('Canonical panel has incomplete development target grid')
        expected['origin'] = fold.origin
        expected['horizon'] = expected.target_day_num - fold.origin
        grid.append(expected[FUTURE_KEYS])
    expected = pd.concat(grid, ignore_index=True)
    for _, frame in forecasts.groupby('model', sort=True):
        validate_predictions(frame, expected)
    scored = attach_actuals(forecasts, panel)
    labels = history_labels(panel, cfg)
    scored = scored.merge(labels, on=['series_id', 'origin'], validate='many_to_one')
    if scored.history_observed.isna().any():
        raise ValueError('Missing training history for evaluated series')
    scored['horizon_week'] = ((scored.horizon - 1) // 7 + 1).astype(int)
    return scored


def diagnostic_metrics(frame):
    """Reuse shared loss definitions; separate both miss directions.

    Coverage includes equality at the interval endpoints. Quantile calibration
    uses both P(Y < q) and P(Y <= q); for discrete counts the target probability
    may lie between these bounds even for a correctly specified distribution.
    """
    record = score_predictions(frame)
    observed = frame.loc[frame.actual.notna()]
    has_intervals = frame[['q10', 'q50', 'q90']].notna().all().all()
    record.update(n_intervals=len(observed) if has_intervals else 0,
                  coverage_gap_80=record['coverage_80'] - .8,
                  lower_miss_rate=np.nan, upper_miss_rate=np.nan)
    if has_intervals and len(observed):
        record['lower_miss_rate'] = float(observed.actual.lt(observed.q10).mean())
        record['upper_miss_rate'] = float(observed.actual.gt(observed.q90).mean())
    for q in ['q10', 'q50', 'q90']:
        record[f'{q}_cdf_lt'] = float(observed.actual.lt(observed[q]).mean()) if has_intervals and len(observed) else np.nan
        record[f'{q}_cdf_le'] = float(observed.actual.le(observed[q]).mean()) if has_intervals and len(observed) else np.nan
    return record


def analyze_uncertainty(predictions, panel, cfg):
    """Return auditable diagnostic tables without training or changing models."""
    scored = prepare_scored(predictions, panel, cfg)
    tables = {}
    for name, keys in {
        'overall': ['model'], 'by_origin': ['model', 'origin'],
        'by_horizon': ['model', 'horizon'], 'by_week': ['model', 'horizon_week'],
        'by_volume': ['model', 'volume_group'],
        'by_volume_week': ['model', 'volume_group', 'horizon_week'],
        'by_series': ['model', 'series_id'],
    }.items():
        records = []
        for values, frame in scored.groupby(keys, observed=True, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            records.append({**dict(zip(keys, values)), **diagnostic_metrics(frame)})
        tables[name] = pd.DataFrame(records)
    # Exact matched keys and canonical actuals were checked before aggregation.
    # Retain ties explicitly; no arbitrary winner or significance claim.
    pairs = []
    sku = tables['by_series']
    for left, right in combinations(sorted(sku.model.unique()), 2):
        a = sku.loc[sku.model.eq(left)].set_index('series_id')
        b = sku.loc[sku.model.eq(right)].set_index('series_id')
        for series_id in a.index:
            delta = a.loc[series_id, 'mae'] - b.loc[series_id, 'mae']
            winner = ('unobserved' if pd.isna(delta) else 'tie' if np.isclose(delta, 0, atol=1e-12, rtol=0)
                      else left if delta < 0 else right)
            pairs.append(dict(series_id=series_id, model_a=left, model_b=right,
                              mae_a=a.loc[series_id, 'mae'], mae_b=b.loc[series_id, 'mae'],
                              mae_difference_a_minus_b=delta, winner=winner))
    tables['paired_series'] = pd.DataFrame(pairs, columns=[
        'series_id', 'model_a', 'model_b', 'mae_a', 'mae_b', 'mae_difference_a_minus_b', 'winner'])
    tables['history_labels'] = scored[['series_id', 'origin', 'volume_group', 'history_mean',
                                      'history_observed', 'history_zero_fraction', 'trailing_zero_days']].drop_duplicates()
    tables['worst_series'] = sku.sort_values(['model', 'mae'], ascending=[True, False]).groupby(
        'model', sort=True).head(10).reset_index(drop=True)
    return tables


def plot_diagnostics(tables, output_dir):
    """Export calibration plots with point-only models excluded from intervals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for model, frame in tables['by_week'].groupby('model', sort=True):
        if not frame.n_intervals.sum():
            continue
        axes[0].plot(frame.horizon_week, frame.coverage_80, marker='o', label=model)
        axes[1].plot(frame.horizon_week, frame.mean_interval_width_80, marker='o', label=model)
        axes[2].plot(frame.horizon_week, frame.upper_miss_rate, marker='o', label=model)
    axes[0].axhline(.8, color='black', linestyle='--', label='Nominal 80%')
    axes[2].axhline(.1, color='black', linestyle='--', label='Nominal upper tail 10%')
    for axis, title, ylabel in zip(axes, ['Interval coverage', 'Interval width', 'Upper-tail misses'],
                                   ['Fraction (inclusive endpoints)', 'Units sold', 'Fraction above P90']):
        axis.set(title=title, xlabel='Forecast week', ylabel=ylabel)
        axis.legend(fontsize=6)
        axis.grid(alpha=.2)
    axes[0].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)
    fig.savefig(output_dir / 'calibration_by_week.png', dpi=150)
    plt.close(fig)
    frame = tables['by_volume'].loc[lambda x: x.n_intervals.gt(0)]
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if not frame.empty:
        frame.pivot(index='volume_group', columns='model', values='coverage_80').reindex(
            ['low', 'medium', 'high']).plot.bar(ax=axis)
    axis.axhline(.8, color='black', linestyle='--')
    axis.set(title='P10–P90 coverage by training-history volume', ylabel='Empirical coverage', ylim=(0, 1))
    fig.savefig(output_dir / 'coverage_by_volume.png', dpi=150)
    plt.close(fig)
