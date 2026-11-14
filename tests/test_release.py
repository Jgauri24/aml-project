"""Final holdout isolation, one-shot scoring and saved artifact guarantees."""
import json
import pandas as pd
import pytest

from src.release import evaluate as release
from src.evaluation.backtesting import DummyForecaster


def test_final_evaluation_is_once_and_models_see_only_history(tiny_panel, tmp_path, monkeypatch):
    panel, cfg = tiny_panel
    seen = []
    # Use the existing serializable dummy model for on-disk artifact output.
    def constructor(dest):
        return DummyForecaster(1)
    root = tmp_path / 'release'
    root.mkdir()
    (root / 'freeze.json').write_text('{}')
    config = tmp_path / 'models.yaml'
    config.write_text('primary_model: seasonal_naive\nmodels: {}\n')
    monkeypatch.setattr(release, 'RELEASE', root)
    monkeypatch.setattr(release, 'FROZEN', config)
    monkeypatch.setattr(release, 'verify_freeze', lambda: {})
    monkeypatch.setattr(release, 'load_config', lambda: cfg)
    monkeypatch.setattr(release, 'load_panel', lambda *args, **kwargs: panel)
    monkeypatch.setattr(release, 'factories', lambda _: {'seasonal_naive': constructor})
    original = release.construct_inputs
    def construct(p, fold):
        inputs = original(p, fold)
        seen.append((inputs.history.day_num.max(), set(inputs.future_known.columns)))
        return inputs
    monkeypatch.setattr(release, 'construct_inputs', construct)
    release.evaluate('final')
    assert seen[0][0] == cfg['forecast']['final_origin']
    assert 'units_sold' not in seen[0][1]
    prediction = pd.read_parquet(root / 'final/predictions.parquet')
    assert prediction.target_day_num.min() == 13
    assert prediction.target_day_num.max() == 15
    assert len(prediction) == 9
    receipt = json.loads((root / 'final/receipt.json').read_text())
    assert receipt['final_holdout_evaluated'] is True
    with pytest.raises(FileExistsError):
        release.evaluate('final')
    assert len(seen) == 1


def test_started_scoring_cannot_be_retried(tmp_path, monkeypatch):
    (tmp_path / 'final').mkdir()
    (tmp_path / 'final/EVALUATION_STARTED.json').write_text('{}')
    monkeypatch.setattr(release, 'RELEASE', tmp_path)
    monkeypatch.setattr(release, 'verify_freeze', lambda: {})
    with pytest.raises(FileExistsError, match='already started'):
        release.evaluate('final')
