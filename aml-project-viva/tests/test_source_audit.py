"""The source audit uses the four analytical inputs also required by readiness."""
from pathlib import Path

from src.data.load_data import EXPECTED_FILES
from src.data.understanding import audit_raw


def test_audit_and_dictionary_use_only_the_four_analysis_sources(tiny_project):
    cfg, raw = tiny_project
    expected = {"calendar.csv", "sell_prices.csv", "sales_train_validation.csv",
                "sales_train_evaluation.csv"}
    assert set(EXPECTED_FILES) == expected
    assert {path.name for path in raw.glob("*.csv")} == expected
    audit = audit_raw(cfg)
    assert set(audit["files"]) == expected
    assert audit["overlap_check"]["matched"]
    dictionary = (Path(cfg["paths"]["reports"]) / "dataset_dictionary.md").read_text()
    for name in expected:
        assert f"## {name}" in dictionary
