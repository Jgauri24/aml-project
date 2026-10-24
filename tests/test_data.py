from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
from src.config import load_config
from src.data.load_data import load_sales, load_calendar, load_prices, assert_unique
from src.data.select_skus import select_series
from src.data.preprocess import build_panel, add_quality_flags, run_preprocessing
from src.data.validate import validate_panel


def test_sales_filter_before_reshape_and_end_day(tiny_project):
    _, raw = tiny_project
    all_sales = load_sales(raw, 6)
    sales = load_sales(raw, 6, all_sales.iloc[:1])
    assert len(sales) == 1 and "d_7" not in sales and sales.d_1.dtype == "float32"


def test_missing_required_day_rejected(tiny_project):
    _, raw = tiny_project
    with pytest.raises(ValueError, match="missing columns"):
        load_sales(raw, 16)


def test_unknown_selected_series_rejected(tiny_project):
    _, raw = tiny_project
    with pytest.raises(ValueError, match="No sales series"):
        load_sales(raw, 6, pd.DataFrame({"item_id": ["absent"], "store_id": ["CA_1"]}))


@pytest.mark.parametrize("kind", ["sales", "calendar", "prices"])
def test_duplicate_raw_keys_rejected(tiny_project, kind):
    _, raw = tiny_project
    name, reader = {"sales": ("sales_train_evaluation.csv", lambda: load_sales(raw, 6)),
                    "calendar": ("calendar.csv", lambda: load_calendar(raw)),
                    "prices": ("sell_prices.csv", lambda: load_prices(raw))}[kind]
    df = pd.read_csv(raw / name)
    pd.concat([df, df.iloc[:1]]).to_csv(raw / name, index=False)
    with pytest.raises(ValueError, match="duplicate key"):
        reader()


def test_selection_deterministic_sparse_and_future_safe(tiny_project):
    cfg, raw = tiny_project
    sales = load_sales(raw, 15)
    first, report = select_series(sales, cfg)
    mutated = sales.copy()
    mutated[[f"d_{d}" for d in range(7, 16)]] = 1e6
    second, _ = select_series(mutated.sample(frac=1, random_state=1), cfg)
    pd.testing.assert_frame_equal(first, second)
    assert first.volume_group.value_counts().to_dict() == {"high": 1, "low": 1, "medium": 1}
    assert first.loc[first.item_id.eq("A"), "zero_fraction"].iloc[0] == 1
    assert report["excluded"] == 0


def test_selection_eligibility_explicit(tiny_project):
    cfg, raw = tiny_project
    sales = load_sales(raw, 6)
    sales.loc[0, "d_3"] = np.nan
    with pytest.raises(ValueError, match="eligible series"):
        select_series(sales, cfg)


def test_missing_targets_are_not_zeros_or_filled(tiny_project):
    cfg, raw = tiny_project
    sales = load_sales(raw, 15)
    sales.loc[0, "d_2"] = np.nan
    panel = build_panel(sales, load_calendar(raw), load_prices(raw), 1, 15)
    row = panel.loc[panel.item_id.eq("A") & panel.day_num.eq(2)].iloc[0]
    assert pd.isna(row.units_sold) and pd.isna(row.is_zero_sale)
    assert row.price_missing and row.availability_unknown
    assert validate_panel(panel, 1, 15, 3)["missing_targets"] == 1


def test_prelaunch_flags_are_past_only_and_uncertain(tiny_panel):
    panel, _ = tiny_panel
    a = panel.loc[panel.item_id.eq("A")]
    assert a.loc[a.day_num.le(7), "possible_prelaunch"].all()
    assert not a.loc[a.day_num.ge(8), "possible_prelaunch"].any()
    future_mutated = panel.copy()
    future_mutated.loc[future_mutated.day_num.gt(6), "units_sold"] = 9999
    future_mutated.loc[future_mutated.day_num.gt(6), "sell_price"] = np.nan
    again = add_quality_flags(future_mutated)
    pd.testing.assert_frame_equal(panel.loc[panel.day_num.le(6), ["possible_prelaunch", "price_missing", "is_zero_sale"]],
                                  again.loc[again.day_num.le(6), ["possible_prelaunch", "price_missing", "is_zero_sale"]])


@pytest.mark.parametrize("problem", ["missing_row", "duplicate", "negative", "infinite", "static", "bad_date", "bad_day", "bad_zero_flag"])
def test_panel_validation_rejects_corruption(tiny_panel, problem):
    panel, _ = tiny_panel
    changed = panel.copy()
    if problem == "missing_row": changed = changed.iloc[1:]
    if problem == "duplicate": changed = pd.concat([changed, changed.iloc[:1]])
    if problem == "negative": changed.loc[0, "units_sold"] = -1
    if problem == "infinite": changed.loc[0, "units_sold"] = np.inf
    if problem == "static": changed.loc[0, "state_id"] = "TX"
    if problem == "bad_date": changed.loc[0, "date"] += pd.Timedelta(days=2)
    if problem == "bad_day": changed.loc[0, "d"] = "d_2"
    if problem == "bad_zero_flag": changed.loc[0, "is_zero_sale"] = False
    with pytest.raises(ValueError):
        validate_panel(changed, 1, 15, 3)


def test_join_cardinality_and_unmatched_calendar(tiny_project):
    _, raw = tiny_project
    sales, calendar, prices = load_sales(raw, 15), load_calendar(raw), load_prices(raw)
    with pytest.raises(pd.errors.MergeError):
        build_panel(sales, calendar, pd.concat([prices, prices.iloc[:1]]), 1, 15)
    with pytest.raises(ValueError, match="Unmatched calendar"):
        build_panel(sales, calendar.iloc[1:], prices, 1, 15)


def test_full_pipeline_roundtrip(tiny_project):
    cfg, raw = tiny_project
    before = {p.name: p.read_bytes() for p in raw.glob("*.csv")}
    panel, metadata = run_preprocessing(cfg)
    restored = pd.read_parquet(cfg["paths"]["processed"] + "/demand_panel.parquet")
    # Arrow may change only categorical dictionary string storage (NA vs NaN backend).
    # Check values/missingness and logical column dtypes, rather than that backend.
    pd.testing.assert_frame_equal(panel, restored, check_categorical=False)
    assert panel.dtypes.astype(str).to_dict() == restored.dtypes.astype(str).to_dict()
    assert metadata["validation"]["rows"] == 45
    assert all(p.read_bytes() == before[p.name] for p in raw.glob("*.csv"))


def test_stale_artifacts_rejected(tiny_project):
    from src.data.preprocess import load_panel
    cfg, _ = tiny_project
    run_preprocessing(cfg)
    assert load_panel(cfg, cutoff=6).day_num.max() == 6
    changed_config = deepcopy(cfg)
    changed_config["seed"] = 43
    with pytest.raises(ValueError, match="configuration differs"):
        load_panel(changed_config)
    selected_path = cfg["paths"]["interim"] + "/selected_skus.csv"
    selected = pd.read_csv(selected_path)
    selected.iloc[::-1].to_csv(selected_path, index=False)
    with pytest.raises(ValueError, match="Selected series file changed"):
        load_panel(cfg)


@pytest.mark.parametrize("source", ["sales", "price"])
def test_raw_invalid_numeric_domains_rejected(tiny_project, source):
    _, raw = tiny_project
    if source == "sales":
        path = raw / "sales_train_evaluation.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "d_1"] = -1
        reader = lambda: load_sales(raw, 6)
    else:
        path = raw / "sell_prices.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "sell_price"] = 0
        reader = lambda: load_prices(raw)
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError): reader()
