"""Direct training must be indistinguishable from historical-origin inference."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtesting import ForecastInputs, FUTURE_KEYS
from src.features.feature_engineering import FeatureEngineer, _price_features_train, _price_snapshot_at_origin


@pytest.fixture
def long_inputs():
    rows, static = [], []
    for offset, sid in enumerate(['A', 'B']):
        metadata = dict(series_id=sid, item_id=sid, store_id='STORE', dept_id='DEPT', cat_id='CAT', state_id='CA')
        static.append(metadata)
        for day in range(1, 85):
            date = pd.Timestamp('2020-01-01') + pd.Timedelta(days=day-1)
            rows.append({**metadata, 'day_num': day, 'date': date,
                         'units_sold': float(day + 100 * offset),
                         'sell_price': np.nan if 15 <= day <= 21 else float(1 + day // 10),
                         'snap': day % 2, 'month': date.month, 'year': date.year,
                         'event_name_1': 'Holiday' if day % 20 == 0 else None, 'event_name_2': pd.NA})
    history = pd.DataFrame(rows)
    future = pd.DataFrame([{'series_id': sid, 'origin': 84, 'horizon': h, 'target_day_num': 84+h,
                           'target_date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=83+h),
                           'month': 4, 'year': 2020, 'event_name_1': None, 'event_name_2': None}
                          for sid in ['A', 'B'] for h in range(1,29)])
    return ForecastInputs(history, pd.DataFrame(static), future, 84, 28)


def test_direct_labels_keys_and_lag_availability(long_inputs):
    builder = FeatureEngineer().fit(long_inputs)
    X, y, keys = builder.build_direct_train_features(long_inputs)
    assert set(keys.origin) == {28, 56}
    assert set(X.horizon) == set(range(1,29))
    assert len(X) == 112
    assert not keys.duplicated(['series_id','target_day_num']).any()
    lookup = long_inputs.history.set_index(['series_id','day_num']).units_sold
    for i, row in enumerate(keys.itertuples(index=False)):
        assert row.target_day_num <= long_inputs.origin
        assert row.target_day_num == row.origin + row.horizon
        assert y.iloc[i] == lookup.loc[(row.series_id,row.target_day_num)]
        for lag in builder.lags:
            if row.horizon > lag:
                assert np.isnan(X.iloc[i][f'lag_{lag}'])
            else:
                assert X.iloc[i][f'lag_{lag}'] == lookup.loc[(row.series_id,row.target_day_num-lag)]


def test_direct_training_equals_prediction_at_each_historical_origin(long_inputs):
    builder = FeatureEngineer().fit(long_inputs)
    X, _, keys = builder.build_direct_train_features(long_inputs)
    calendar = long_inputs.history.set_index(['series_id','day_num'])
    for origin in keys.origin.unique():
        mask = keys.origin.eq(origin)
        grid = keys.loc[mask].copy().reset_index(drop=True)
        index = pd.MultiIndex.from_arrays([grid.series_id, grid.target_day_num])
        for name in ['month','year','event_name_1','event_name_2']:
            grid[name] = calendar[name].reindex(index).to_numpy()
        historical = ForecastInputs(long_inputs.history.loc[long_inputs.history.day_num.le(origin)].copy(),
                                    long_inputs.static.copy(), grid, int(origin), 28)
        expected = FeatureEngineer().fit(historical).build_predict_features(historical)
        expected['horizon'] = grid.horizon.to_numpy()
        pd.testing.assert_frame_equal(X.loc[mask].reset_index(drop=True), expected, check_dtype=False)


def test_training_origin_snapshot_ignores_later_sales_prices_and_snap(long_inputs):
    before, labels_before, keys = FeatureEngineer().fit(long_inputs).build_direct_train_features(long_inputs)
    changed = deepcopy(long_inputs)
    future = changed.history.day_num.gt(28)
    changed.history.loc[future, ['units_sold','sell_price','snap']] = [99999,9999,9]
    after, labels_after, other_keys = FeatureEngineer().fit(changed).build_direct_train_features(changed)
    mask = keys.origin.eq(28)
    pd.testing.assert_frame_equal(keys, other_keys)
    pd.testing.assert_frame_equal(before.loc[mask], after.loc[mask])
    assert not labels_before.loc[mask].equals(labels_after.loc[mask])


def test_price_ratio_uses_calendar_days_and_matches_one_step_snapshot():
    frame = pd.DataFrame({'series_id':['A']*16,'day_num':range(1,17),
                          'sell_price':[1.0]+[np.nan]*13+[2.0,999.0]})
    snapshot = _price_snapshot_at_origin(frame,15)
    assert snapshot == {'last_known_price':2.0,'price_change_ratio':2.0}
    train = _price_features_train(frame).iloc[-1]
    assert train.last_known_price == snapshot['last_known_price']
    assert train.price_change_ratio == snapshot['price_change_ratio']


def test_future_unknown_columns_are_ignored_and_row_order_is_preserved(long_inputs):
    builder = FeatureEngineer().fit(long_inputs)
    original = builder.build_predict_features(long_inputs)
    changed = deepcopy(long_inputs)
    changed.future_known = changed.future_known.sample(frac=1,random_state=42)
    order = changed.future_known.index.to_numpy()
    for name in ['units_sold','sell_price','snap','last_known_price','rmean_7']:
        changed.future_known[name] = 99999
    result = builder.build_predict_features(changed)
    pd.testing.assert_frame_equal(result, original.iloc[order].reset_index(drop=True))
    assert result.snap.eq(0).all()  # historical SNAP at origin 84, not injected future value


def test_feature_state_json_roundtrip_and_missing_targets(long_inputs):
    long_inputs.history.loc[long_inputs.history.day_num.eq(35),'units_sold'] = np.nan
    builder = FeatureEngineer().fit(long_inputs)
    X, y, keys = builder.build_direct_train_features(long_inputs)
    assert y.notna().all() and len(X)==110
    assert not keys.target_day_num.eq(35).any()
    encoded = json.dumps(builder.to_state(), allow_nan=False)
    restored = FeatureEngineer.from_state(json.loads(encoded))
    pd.testing.assert_frame_equal(builder.build_predict_features(long_inputs),
                                  restored.build_predict_features(long_inputs))


def test_history_after_origin_is_rejected(long_inputs):
    long_inputs.origin = 83
    with pytest.raises(ValueError,match='stop at the forecast origin'):
        FeatureEngineer().fit(long_inputs)
