"""Read-only dashboard data access; no training or fabricated forecasts."""
from pathlib import Path
import pandas as pd
from src.config import ROOT

RELEASE = ROOT / 'artifacts/release'
NAMES = {'seasonal_naive': 'Weekly seasonal naive', 'kalman': 'Seasonal Kalman',
         'lightgbm': 'LightGBM', 'deepar': 'DeepAR'}
COLORS = {'seasonal_naive': '#83959a', 'kalman': '#168c85', 'lightgbm': '#d47745', 'deepar': '#7863ac'}


def load_bundle(root=RELEASE):
    root = Path(root)
    history_path = root / 'demo_history.parquet'
    if not history_path.is_file():
        return None
    history = pd.read_parquet(history_path)
    segments = pd.read_csv(root / 'segments.csv')
    results = {}
    for scope in ['development', 'final']:
        directory = root / scope
        if (directory / 'scored_predictions.parquet').is_file():
            results[scope] = dict(predictions=pd.read_parquet(directory / 'scored_predictions.parquet'),
                                  metrics=pd.read_csv(directory / 'metrics.csv'),
                                  weeks=pd.read_csv(directory / 'metrics_by_week.csv'),
                                  runtime=pd.read_csv(directory / 'runtime.csv'))
    return history, segments, results


def select_forecasts(predictions, series_id, origin, models):
    return predictions.loc[predictions.series_id.eq(series_id) & predictions.origin.eq(origin)
                           & predictions.model.isin(models)].sort_values(['model', 'horizon']).copy()
