"""Milestone 15 diagnostics; consumes existing predictions, never retrains models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.analysis.uncertainty_analysis import analyze_uncertainty, plot_diagnostics
from src.config import ROOT, config_hash, data_path, load_config
from src.data.preprocess import load_panel


def discover_predictions(processed):
    """Only established model locations; avoid duplicate audit/test artifacts."""
    processed = Path(processed)
    candidates = [processed / 'lightgbm/predictions.parquet',
                  processed / 'deepar/predictions.parquet']
    for directory in ['kalman_forecaster', 'seasonal_naive']:
        candidates.extend(sorted((processed / directory).glob('*/predictions.parquet')))
    return [p for p in candidates if p.is_file()]


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(tables, metadata, output):
    overall = tables['overall']
    columns = ['model', 'n_observed', 'n_intervals', 'mae', 'pinball_q10', 'pinball_q50',
               'pinball_q90', 'coverage_80', 'mean_interval_width_80', 'lower_miss_rate', 'upper_miss_rate']
    table = overall[columns].round(4)
    lines = ['# Milestone 15: uncertainty diagnostics', '',
             'Computed from saved row-level predictions and canonical observed sales. No model was retrained.', '',
             '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for values in table.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join('NA' if pd.isna(v) else str(v) for v in values) + ' |')
    lines += ['', '## Evidence and interpretation', '',
              '- All included models passed the same complete series/origin/horizon/date grid check; saved actuals were replaced by canonical actuals.',
              '- Missing actuals are excluded equally for all models; zeros are observed outcomes. Counts are exported for every group.',
              '- Coverage uses inclusive P10–P90 endpoints. Lower and upper misses separate overprediction from underprediction.',
              '- Count-valued demand can produce coverage above 80% even with valid quantiles. The exported strict/non-strict CDF bounds help interpret ties at zero; excess coverage alone does not prove miscalibration.',
              '- Volume terciles and trailing zero runs are computed from each origin’s training window only. Volume ties break by series ID.',
              '- Paired SKU MAE differences use model A minus model B: negative favors A. Ties remain ties. These are descriptive comparisons, not independent-row significance tests.',
              '- Point-only models retain NA uncertainty metrics. No latent demand or verified stockout labels are available.',
              '- The final holdout is excluded. Shared evaluation keys do not by themselves establish that external checkpoints were trained without leakage; inspect their training provenance.',
              '', '## Available and missing evidence', '',
              'Models: ' + ', '.join(metadata['models']) + '.',
              'Missing standard prediction files: ' + (', '.join(metadata['missing_prediction_files']) or 'none') + '.',
              'DeepAR versus Kalman conclusions require genuine saved DeepAR forecasts on the same complete grid.',
              '', '## Outputs', '',
              '`overall.csv`, `by_origin.csv`, `by_horizon.csv`, `by_week.csv`, `by_volume.csv`,',
              '`by_volume_week.csv`, `by_series.csv`, `paired_series.csv`, `worst_series.csv`,',
              '`history_labels.csv`, calibration plots and a source-hash manifest.', '']
    (output / 'report.md').write_text('\n'.join(lines))


def run_analysis(cfg=None, prediction_paths=None, output_dir=None):
    cfg = load_config() if cfg is None else cfg
    processed = data_path(cfg, 'processed')
    paths = discover_predictions(processed) if prediction_paths is None else [Path(p) for p in prediction_paths]
    if not paths:
        raise FileNotFoundError('No genuine prediction Parquet files found; run the model backtests first')
    panel = load_panel(cfg, cutoff=cfg['forecast']['final_origin'])
    # Match the experiment's optional cohort filters before checking complete grids.
    for key, column in [('sku_ids', 'item_id'), ('store_ids', 'store_id')]:
        values = cfg.get('backtest', {}).get(key)
        if values:
            panel = panel.loc[panel[column].isin(values)]
    predictions = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    tables = analyze_uncertainty(predictions, panel, cfg)
    output = Path(output_dir) if output_dir else ROOT / 'reports/uncertainty_analysis'
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f'{name}.csv', index=False)
    plot_diagnostics(tables, output)
    standard = ['deepar/predictions.parquet', 'lightgbm/predictions.parquet',
                'seasonal_naive/weekly_seasonal/predictions.parquet',
                'kalman_forecaster/local_level_seasonal/predictions.parquet']
    metadata = dict(config_sha256=config_hash(cfg),
                    panel_sha256=_digest(processed / 'demand_panel.parquet'),
                    sources=[dict(path=str(p.resolve()), sha256=_digest(p)) for p in paths],
                    models=sorted(predictions.model.unique()),
                    missing_prediction_files=[p for p in standard if not (processed / p).is_file()],
                    final_holdout_evaluated=False,
                    nominal_interval_coverage=.8,
                    note='Descriptive development diagnostics; training provenance must be audited separately.')
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    write_report(tables, metadata, output)
    return tables, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--prediction', action='append', help='Explicit prediction Parquet path; repeat for each model')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    tables, metadata = run_analysis(load_config(args.config), args.prediction, args.output_dir)
    print(tables['overall'][['model', 'n_intervals', 'coverage_80', 'mean_interval_width_80']].to_string(index=False))
    print('Missing prediction files:', metadata['missing_prediction_files'])


if __name__ == '__main__':
    main()
