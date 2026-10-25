from pathlib import Path
import subprocess
import sys

import pytest

from scripts.smoke_test import check_required_files, run_smoke
from src.data.preprocess import run_preprocessing


def test_smoke_runs_shared_development_engine_without_writes(tiny_project):
    cfg, _ = tiny_project
    run_preprocessing(cfg)
    root = Path(cfg["paths"]["processed"]).parent
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = run_smoke(cfg)
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert result["status"] == "passed"
    assert result["required_files"] == 8
    assert result["prediction_rows"] == 18 and result["max_target_day"] == 12
    assert not result["final_holdout_evaluated"] and not result["output_files_written"]
    assert before == after


def test_smoke_explains_missing_processed_data(tiny_project):
    cfg, _ = tiny_project
    with pytest.raises(FileNotFoundError, match="python -m src.data.preprocess"):
        check_required_files(cfg)


def test_smoke_rejects_missing_raw_source(tiny_project):
    cfg, raw = tiny_project
    run_preprocessing(cfg)
    (raw / "calendar.csv").unlink()
    with pytest.raises(FileNotFoundError, match="calendar.csv"):
        run_smoke(cfg)


def test_smoke_cli_fails_with_useful_message(tiny_project):
    import yaml
    cfg, raw = tiny_project
    config_path = raw.parent / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    result = subprocess.run(
        [sys.executable, "-m", "scripts.smoke_test", "--config", str(config_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "SMOKE FAILED" in result.stderr and "demand_panel.parquet" in result.stderr
