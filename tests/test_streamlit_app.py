"""The real artifact-backed forecast viewer renders and respects selections."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from src.config import ROOT

pytestmark = pytest.mark.skipif(not (ROOT/'artifacts/release/demo_history.parquet').is_file(),
                                reason='Packaged release data is not available')


def test_dashboard_renders_all_research_views():
    app=AppTest.from_file(str(ROOT/'app.py'), default_timeout=45).run()
    assert not app.exception
    assert len(app.tabs)==4
    assert app.selectbox[0].value=='final'
    selected=app.dataframe[0].value
    assert len(selected)==112
    assert set(selected.model)=={'seasonal_naive','kalman','lightgbm','deepar'}


def test_actual_toggle_and_model_filter_change_download_table():
    app=AppTest.from_file(str(ROOT/'app.py'), default_timeout=45).run()
    app.checkbox[1].uncheck().run()
    assert not app.exception
    assert 'actual' not in app.dataframe[0].value.columns
    app.multiselect[0].set_value(['deepar']).run()
    assert not app.exception
    selected=app.dataframe[0].value
    assert len(selected)==28
    assert selected.model.eq('deepar').all()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any('Select at least one model' in v.value for v in app.info)


def test_store_selection_updates_the_sku_cohort():
    app=AppTest.from_file(str(ROOT/'app.py'), default_timeout=45).run()
    stores=app.selectbox(key='store').options
    app.selectbox(key='store').set_value(stores[-1]).run()
    assert not app.exception
    from src.ui.dashboard import load_bundle
    _,segments,_=load_bundle()
    sid=app.selectbox(key='sku').value
    assert segments.set_index('series_id').loc[sid,'store_id']==stores[-1]
