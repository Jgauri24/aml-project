"""A separate, guarded final evaluation; development backtests stay unchanged."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import yaml

from src.config import ROOT, load_config
from src.data.preprocess import load_panel
from src.evaluation.backtesting import construct_inputs, validate_predictions, attach_actuals
from src.evaluation.metrics import score_predictions
from src.evaluation.splits import development_splits, final_holdout_spec

RELEASE = ROOT / 'artifacts/release'
FROZEN = ROOT / 'configs/final_models.yaml'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, data, exclusive=False):
    def convert(value):
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        return value
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(convert(data), stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_hashes():
    files = [ROOT / 'experiment.yaml', FROZEN]
    for folder in ['src/models', 'src/features', 'src/data', 'src/evaluation', 'src/release']:
        files.extend(sorted((ROOT / folder).glob('*.py')))
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def freeze():
    """Record the settings, implementation and dataset before any final scoring."""
    RELEASE.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    load_panel(cfg, cutoff=cfg['forecast']['final_origin'])  # Validate canonical metadata.
    evidence = ROOT / 'reports/deepar_experiments_comparison.csv'
    manifest = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                    sources=source_hashes(),
                    panel_sha256=digest(ROOT / 'data/processed/demand_panel.parquet'),
                    selection_evidence_sha256=digest(evidence),
                    decision='LightGBM primary; seasonal Kalman, weekly naive and hidden-32 DeepAR comparators. No holdout tuning.',
                    final_origin=cfg['forecast']['final_origin'], horizon=cfg['forecast']['horizon'])
    json_write(RELEASE / 'freeze.json', manifest, exclusive=True)
    return manifest


def verify_freeze():
    manifest = json.loads((RELEASE / 'freeze.json').read_text())
    if manifest['sources'] != source_hashes():
        raise ValueError('Frozen implementation/configuration changed; final evaluation refused')
    if manifest['panel_sha256'] != digest(ROOT / 'data/processed/demand_panel.parquet'):
        raise ValueError('Frozen canonical dataset changed')
    return manifest


def factories(settings):
    import torch
    from src.models.seasonal_naive import SeasonalNaiveForecaster
    from src.models.kalman_forecaster import KalmanForecaster
    from src.models.lightgbm_forecaster import LightGBMForecaster
    from src.models.deepar_config import DeepARConfig
    from src.models.deepar_forecaster import DeepARForecaster
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    values = settings['models']
    return dict(seasonal_naive=lambda dest: SeasonalNaiveForecaster(**values['seasonal_naive']),
                kalman=lambda dest: KalmanForecaster(**values['kalman']),
                lightgbm=lambda dest: LightGBMForecaster(values['lightgbm']),
                deepar=lambda dest: DeepARForecaster(DeepARConfig.from_dict(values['deepar']), checkpoint_path=dest))


def run_fold(panel, fold, constructors, output):
    """Model fitting sees explicit history/calendar projections only."""
    inputs = construct_inputs(panel, fold)
    predictions, costs = [], []
    for name, factory in constructors.items():
        dest = output / 'models' / f'origin_{fold.origin}' / name
        dest.mkdir(parents=True, exist_ok=True)
        print(f'Fitting {name} at d_{fold.origin}', flush=True)
        model = factory(dest)
        started = perf_counter()
        model.fit(inputs)
        fit_seconds = perf_counter() - started
        started = perf_counter()
        prediction = model.predict(inputs)
        predict_seconds = perf_counter() - started
        validate_predictions(prediction, inputs.future_known)
        prediction['model'] = name
        if name == 'lightgbm':
            model.save(dest)
        elif name == 'deepar':
            json_write(dest / 'encoder.json', dict(mappings=model.prepared_data_.encoder.mappings_,
                scales=model.prepared_data_.inference_data['scales'].tolist(),
                series_ids=model.prepared_data_.series_ids))
            pd.DataFrame(model.trainer_.history).to_csv(dest / 'training_history.csv', index=False)
        else:
            joblib.dump(model, dest / 'model.joblib', compress=3)
        prediction.to_parquet(dest / 'predictions.parquet', index=False)
        predictions.append(prediction)
        costs.append(dict(model=name, origin=fold.origin, fit_seconds=fit_seconds,
                          predict_seconds=predict_seconds, prediction_rows=len(prediction)))
    return pd.concat(predictions, ignore_index=True), costs


def summarize(scored):
    rows = []
    for model, frame in scored.groupby('model', sort=True):
        rows.append(dict(model=model, **score_predictions(frame)))
    return pd.DataFrame(rows)


def evaluate(mode):
    if mode not in {'development', 'final'}:
        raise ValueError('Mode must be development or final')
    manifest = verify_freeze()
    cfg = load_config()
    output = RELEASE / mode
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'receipt.json').exists():
        raise FileExistsError(f'{mode} already completed; reuse the saved predictions and metrics')
    if mode == 'final' and (output / 'EVALUATION_STARTED.json').exists():
        raise FileExistsError('Final scoring already started; automatic rescoring is prohibited')
    settings = yaml.safe_load(FROZEN.read_text())
    constructors = factories(settings)
    # Development never loads reserved target days. Final inputs use the same
    # allowlists, and target sales are attached only after all forecasts exist.
    panel = load_panel(cfg, cutoff=cfg['forecast']['final_origin'] if mode == 'development' else None)
    folds = development_splits(cfg) if mode == 'development' else [final_holdout_spec(cfg)]
    frames, costs = [], []
    for fold in folds:
        prediction, timing = run_fold(panel, fold, constructors, output)
        frames.append(prediction)
        costs.extend(timing)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_parquet(output / 'predictions.parquet', index=False)
    # All models and settings have been frozen, trained and saved before the
    # evaluator attaches a single holdout target. Exclusive creation guards
    # against accidental retries/concurrent evaluation in this release.
    if mode == 'final':
        json_write(output / 'EVALUATION_STARTED.json',
                   dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                        freeze_sha256=digest(RELEASE / 'freeze.json')), exclusive=True)
    scored = attach_actuals(predictions, panel)
    scored.to_parquet(output / 'scored_predictions.parquet', index=False)
    table = summarize(scored)
    table.to_csv(output / 'metrics.csv', index=False)
    weeks = []
    for (model, week), frame in scored.assign(week=(scored.horizon - 1)//7 + 1).groupby(['model','week']):
        weeks.append(dict(model=model, week=int(week), **score_predictions(frame)))
    pd.DataFrame(weeks).to_csv(output / 'metrics_by_week.csv', index=False)
    pd.DataFrame(costs).to_csv(output / 'runtime.csv', index=False)
    json_write(output / 'receipt.json', dict(mode=mode, completed_at_utc=datetime.now(timezone.utc).isoformat(),
               freeze_sha256=digest(RELEASE / 'freeze.json'), prediction_sha256=digest(output / 'predictions.parquet'),
               final_holdout_evaluated=mode=='final',
               origins=[f.origin for f in folds], max_target_day=int(scored.target_day_num.max()),
               predictions_per_model=len(predictions)//len(constructors),
               no_post_holdout_tuning=True, primary_model=settings['primary_model']))
    print(table[['model','mae','rmse','wape','coverage_80']].to_string(index=False), flush=True)
    return table
