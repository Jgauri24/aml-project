"""Run: python -m streamlit run app.py. Reads genuine packaged model forecasts."""
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.ui.dashboard import load_bundle, select_forecasts, NAMES, COLORS, RELEASE

st.set_page_config(page_title='Retail Demand Observatory', page_icon='📦', layout='wide')
st.markdown('''<style>
.block-container {padding-top:2rem; max-width:1450px;}
h1 {letter-spacing:-1.5px;} [data-testid="stMetric"] {background:#eef3f1;padding:14px;border-radius:8px;}
</style>''', unsafe_allow_html=True)
st.caption('RETAIL DEMAND OBSERVATORY · M5 FORECASTING')
st.title('Every product has a different rhythm')
st.write('Explore observed demand, compare four forecasting approaches, and inspect their uncertainty over 28 days.')


@st.cache_data
def data():
    return load_bundle()


bundle = data()
if bundle is None:
    st.info('Packaged forecast data is missing. Follow docs/REPRODUCING.md to build the release bundle.')
    st.stop()
history, segments, results = bundle
if not results:
    st.info('No evaluated forecast artifacts are available yet.')
    st.stop()

with st.sidebar:
    st.header('Explore forecasts')
    scope = st.selectbox('Evaluation window', list(results),
                         index=list(results).index('final') if 'final' in results else 0,
                         format_func=lambda x: 'Final holdout · 25 Apr–22 May 2016' if x == 'final' else 'Development · four rolling origins')
    store = st.selectbox('Store', sorted(segments.store_id.unique()), key='store')
    choices = segments.loc[segments.store_id.eq(store)].sort_values('item_id')
    sid = st.selectbox('Product / SKU', choices.series_id.tolist(),
                      format_func=lambda x: choices.set_index('series_id').loc[x, 'item_id'], key='sku')
    predictions = results[scope]['predictions']
    origin = st.selectbox('Forecast origin', sorted(predictions.origin.unique()), key='origin')
    models = st.multiselect('Models', list(NAMES), default=list(NAMES), format_func=NAMES.get)
    history_days = st.slider('Historical days shown', 28, 112, 84, step=7)
    interval = st.checkbox('Show P10–P90 intervals', True)
    actuals = st.checkbox('Show observed evaluation sales', True)
    st.caption('Saved research forecasts. The viewer performs no model fitting or parameter selection.')
    st.markdown('[Reproduction guide](https://github.com/Jgauri24/aml-project/blob/main/docs/REPRODUCING.md)')

row = choices.set_index('series_id').loc[sid]
selected = select_forecasts(predictions, sid, origin, models)
tabs = st.tabs(['Forecast explorer', 'Model benchmarks', 'Demand segments', 'Research protocol'])

with tabs[0]:
    st.subheader(f'{row.item_id} / {store}')
    st.caption(f"Training volume: {row.volume_group} · Pattern: {row.demand_segment} · Segment labels use history through d_1801")
    a, b, c = st.columns(3)
    a.metric('Historical mean / day', f'{row.history_mean:.2f}')
    b.metric('Zero-sales days in history', f'{100*row.history_zero_fraction:.1f}%')
    c.metric('Forecast horizon', '28 days')
    if not models:
        st.info('Select at least one model to display its forecasts.')
    else:
        fig, ax = plt.subplots(figsize=(13, 4.6), constrained_layout=True)
        past = history.loc[history.series_id.eq(sid) & history.day_num.between(origin-history_days+1, origin)]
        ax.plot(past.date, past.units_sold, color='#193d47', linewidth=1.4, label='Historical sales')
        for model, frame in selected.groupby('model', sort=True):
            ax.plot(frame.target_date, frame.y_pred, color=COLORS[model], linewidth=1.8, label=NAMES[model])
            if interval and frame.q10.notna().all():
                ax.fill_between(frame.target_date, frame.q10, frame.q90, color=COLORS[model], alpha=.12,
                                label=f'{NAMES[model]} P10–P90')
        if actuals and not selected.empty:
            observed = selected.drop_duplicates('target_day_num').sort_values('target_day_num')
            ax.plot(observed.target_date, observed.actual, color='#111111', linewidth=1.2, linestyle=':', label='Observed evaluation sales')
        if not selected.empty:
            ax.axvline(selected.target_date.min(), color='#9eaeb1', linestyle='--', linewidth=1)
        ax.set(ylabel='Observed units sold', xlabel='Date', ylim=(0, None))
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.legend(loc='upper left', fontsize=8, ncol=3)
        st.pyplot(fig)
        plt.close(fig)
        st.caption('P10–P90 bands are available for Kalman and DeepAR. LightGBM and seasonal naive provide point forecasts only.')
        columns = ['model', 'target_date', 'horizon', 'y_pred', 'q10', 'q50', 'q90']
        if actuals:
            columns += ['actual']
        st.dataframe(selected[columns].rename(columns={'y_pred':'point_forecast'}), hide_index=True, use_container_width=True)
        st.download_button('Download selected forecasts', selected[columns].to_csv(index=False),
                           file_name=f'{sid}_{scope}_forecasts.csv', mime='text/csv')

with tabs[1]:
    st.subheader('One shared test grid')
    metrics = results[scope]['metrics']
    cols = ['model','n_observed','mae','rmse','wape','pinball_q10','pinball_q50','pinball_q90','coverage_80','mean_interval_width_80']
    st.dataframe(metrics[cols].round(4), hide_index=True, use_container_width=True)
    st.caption('MAE, RMSE and width are in units sold. WAPE is a ratio. NA indicates unavailable quantiles. Values pool the complete selected cohort, not only the displayed SKU.')
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
        ordered = metrics.sort_values('mae')
        ax.barh([NAMES[m] for m in ordered.model], ordered.mae, color=[COLORS[m] for m in ordered.model])
        ax.invert_yaxis()
        ax.set(xlabel='MAE (units)', title='Average forecast error')
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
        for model, frame in results[scope]['weeks'].groupby('model'):
            ax.plot(frame.week, frame.wape, marker='o', color=COLORS[model], label=NAMES[model])
        ax.set(xlabel='Forecast week', ylabel='WAPE (ratio)', title='Error over the forecast horizon', xticks=[1,2,3,4])
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    st.subheader('Measured model costs')
    timing = results[scope]['runtime'].groupby('model')[['fit_seconds','predict_seconds']].mean().reset_index()
    st.dataframe(timing.round(3), hide_index=True, use_container_width=True)
    st.caption('Mean seconds per origin on this machine. PyTorch/LightGBM: one thread; Kalman: four fit workers. Training time includes model-specific preparation; prediction time excludes shared scoring.')

with tabs[2]:
    st.subheader('Sparse demand is common across the cohort')
    st.caption('Labels use observed training sales through d_1801. Dense: <10% zeros. Intermittent: 10–<50%. Sparse: ≥50%. These descriptive labels do not identify stockouts.')
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
        counts = segments.demand_segment.value_counts()
        ax.bar(counts.index, counts.values, color='#168c85')
        ax.set(ylabel='Product-store series', title='Demand frequency groups')
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
        for group, frame in segments.groupby('volume_group'):
            ax.scatter(frame.history_mean, frame.history_zero_fraction, s=22, alpha=.65, label=group)
        ax.set(xscale='symlog', xlabel='Mean historical units / day', ylabel='Fraction of zero-sales days', title='Volume and intermittency')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    merged = predictions.merge(segments[['series_id','demand_segment']], on='series_id', validate='many_to_one')
    observed = merged.loc[merged.actual.notna()].copy()
    observed['absolute_error'] = (observed.actual-observed.y_pred).abs()
    grouped = observed.groupby(['model','demand_segment']).absolute_error.mean().unstack('model')
    st.write('MAE by demand segment')
    st.dataframe(grouped.round(4), use_container_width=True)
    st.download_button('Download segment metadata', segments.to_csv(index=False), 'demand_segments.csv', 'text/csv')

with tabs[3]:
    st.subheader('Frozen experiment, recorded evidence')
    st.write('240 product-store series. Four development origins: 1801, 1829, 1857 and 1885. Final origin: 1913. Forecast length: 28 days.')
    st.write('The final release freezes model settings and source hashes before attaching holdout targets. The final benchmark is a single historical holdout, not a live future forecast.')
    st.write('DeepAR uses a custom PyTorch LSTM with a negative-binomial likelihood, 56-day context and 200 sampled trajectories. Kalman supplies Gaussian intervals clipped at zero.')
    st.write('Observed sales can be censored by availability. Zero sales do not prove stockouts, and these forecasts do not estimate unobserved lost demand.')
    if (RELEASE / 'final/receipt.json').is_file():
        st.json(json.loads((RELEASE / 'final/receipt.json').read_text()))
