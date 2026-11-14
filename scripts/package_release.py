"""Build the read-only demo bundle and figures from recorded evaluation outputs."""
from __future__ import annotations
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.data.preprocess import load_panel
from src.analysis.uncertainty_analysis import history_labels, diagnostic_metrics
from src.release.evaluate import RELEASE, digest, json_write

COLORS = {'seasonal_naive':'#83959a', 'kalman':'#168c85', 'lightgbm':'#d47745', 'deepar':'#7863ac'}
NAMES = {'seasonal_naive':'Weekly naive','kalman':'Seasonal Kalman','lightgbm':'LightGBM','deepar':'DeepAR'}
ORDER = list(NAMES)
FIG = ROOT / 'artifacts/paper/figures'


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', dpi=190, bbox_inches='tight')
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    panel = load_panel(cfg, cutoff=cfg['forecast']['final_origin'])
    labels = history_labels(panel, cfg).query('origin == 1801').drop(columns='origin')
    identities = panel[['series_id','item_id','store_id','cat_id','dept_id','state_id']].drop_duplicates()
    segments = identities.merge(labels, on='series_id', validate='one_to_one')
    segments['demand_segment'] = pd.cut(segments.history_zero_fraction, [-.001,.1,.5,1.],
        labels=['dense','intermittent','sparse'], right=False).astype('string').fillna('sparse')
    segments.to_csv(RELEASE / 'segments.csv', index=False)
    panel.loc[panel.day_num.ge(1690), ['series_id','item_id','store_id','day_num','date','units_sold']].to_parquet(
        RELEASE / 'demo_history.parquet', index=False)
    all_metrics, conditional, pairs, data = [], [], [], {}
    for scope in ['development','final']:
        dest = RELEASE / scope
        m = pd.read_csv(dest / 'metrics.csv')
        m['scope'] = scope
        all_metrics.append(m)
        p = pd.read_parquet(dest / 'scored_predictions.parquet')
        p = p.merge(segments[['series_id','volume_group','demand_segment']], on='series_id', validate='many_to_one')
        data[scope] = dict(metrics=m, predictions=p, weeks=pd.read_csv(dest/'metrics_by_week.csv'), runtime=pd.read_csv(dest/'runtime.csv'))
        for keys in [['model','volume_group'],['model','demand_segment']]:
            for values, frame in p.groupby(keys, observed=True):
                conditional.append(dict(scope=scope, grouping=keys[1], group=values[1], model=values[0], **diagnostic_metrics(frame)))
        per = p.assign(ae=(p.actual-p.y_pred).abs()).groupby(['series_id','model']).ae.mean().unstack('model')
        for sid, row in per.iterrows():
            delta = row.kalman-row.deepar
            pairs.append(dict(scope=scope, series_id=sid, kalman_mae=row.kalman, deepar_mae=row.deepar,
                              winner='tie' if np.isclose(delta,0,atol=1e-12,rtol=0) else 'kalman' if delta<0 else 'deepar'))
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(RELEASE / 'benchmark.csv', index=False)
    pd.DataFrame(conditional).to_csv(RELEASE / 'conditional_metrics.csv', index=False)
    pd.DataFrame(pairs).to_csv(RELEASE / 'kalman_vs_deepar.csv', index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    # Scope-separated point accuracy: no direct historical improvement claim.
    fig, axes = plt.subplots(1, 2, figsize=(10,3.1), constrained_layout=True)
    for ax, scope in zip(axes, ['development','final']):
        table = data[scope]['metrics'].set_index('model').reindex(ORDER)
        ax.bar(np.arange(4), table.wape, color=[COLORS[m] for m in ORDER])
        ax.set_xticks(range(4), [NAMES[m] for m in ORDER], rotation=18, ha='right')
        ax.set(title='Four development origins' if scope=='development' else 'Single final holdout', ylabel='WAPE (ratio)')
        for i,v in enumerate(table.wape): ax.text(i,v+.015,f'{v:.3f}',ha='center',fontsize=9)
        ax.set_ylim(0,table.wape.max()*1.2)
    save(fig,'benchmark')
    fig, axes = plt.subplots(1,2,figsize=(10,3),constrained_layout=True)
    for ax, scope in zip(axes,['development','final']):
        for model, frame in data[scope]['weeks'].groupby('model'):
            ax.plot(frame.week,frame.wape,marker='o',label=NAMES[model],color=COLORS[model])
        ax.set(title=scope.capitalize(),xlabel='Forecast week',ylabel='WAPE (ratio)',xticks=[1,2,3,4])
        ax.legend(fontsize=7)
    save(fig,'horizon')
    conditional = pd.DataFrame(conditional)
    cal = conditional.query("scope == 'final' and grouping == 'volume_group'")
    fig, axes = plt.subplots(1,2,figsize=(10,3),constrained_layout=True)
    for model in ['kalman','deepar']:
        frame=cal.query('model == @model').set_index('group').reindex(['low','medium','high'])
        axes[0].plot(range(3),frame.coverage_80,marker='o',color=COLORS[model],label=NAMES[model])
        axes[1].plot(range(3),frame.mean_interval_width_80,marker='o',color=COLORS[model],label=NAMES[model])
    axes[0].axhline(.8,color='black',linestyle='--',linewidth=1)
    for ax in axes:
        ax.set_xticks(range(3),['Low','Medium','High']);ax.set_xlabel('Training-volume group');ax.legend(fontsize=8)
    axes[0].set(title='Final interval coverage',ylabel='Fraction covered',ylim=(0,1))
    axes[1].set(title='Final interval width',ylabel='Units sold',ylim=(0,None))
    save(fig,'calibration')
    chosen=[]
    for group in ['low','medium','high']:
        frame=segments.query('volume_group == @group').sort_values(['history_mean','series_id'])
        chosen.append(frame.iloc[len(frame)//2].series_id)
    fig, axes=plt.subplots(3,1,figsize=(10,7),constrained_layout=True)
    examples=[]
    for ax, sid in zip(axes,chosen):
        h=panel.loc[panel.series_id.eq(sid)&panel.day_num.ge(1858)]
        ax.plot(h.date,h.units_sold,color='#193d47',linewidth=1,label='Historical sales')
        subset=data['final']['predictions'].loc[lambda x:x.series_id.eq(sid)]
        for model,frame in subset.groupby('model'):
            ax.plot(frame.target_date,frame.y_pred,color=COLORS[model],label=NAMES[model],linewidth=1.4)
            if model in ['kalman','deepar']:
                ax.fill_between(frame.target_date,frame.q10,frame.q90,color=COLORS[model],alpha=.12)
        actual=subset.drop_duplicates('target_date')
        ax.plot(actual.target_date,actual.actual,color='black',linestyle=':',label='Holdout actual')
        ax.axvline(actual.target_date.min(),color='#aaa',linestyle='--',linewidth=.7)
        ax.set(title=f'{sid} ({segments.set_index("series_id").loc[sid,"volume_group"]} volume)',ylabel='Units sold',ylim=(0,None))
        examples.append(dict(series_id=sid,history=h[['date','units_sold']].astype({'date':str}).to_dict('records'),
                             forecasts=subset.assign(target_date=subset.target_date.astype(str)).replace({np.nan:None}).to_dict('records')))
    axes[0].legend(fontsize=7,ncol=3,loc='upper left')
    save(fig,'forecast_examples')
    fig, axes=plt.subplots(1,2,figsize=(10,3.3),constrained_layout=True)
    for group,frame in segments.groupby('volume_group'):
        axes[0].scatter(frame.history_mean,frame.history_zero_fraction,alpha=.6,s=14,label=group)
    axes[0].set(xscale='symlog',xlabel='Historical mean units/day',ylabel='Zero-sales fraction',title='Training demand structure')
    axes[0].legend(fontsize=8)
    segmented=conditional.query("scope == 'final' and grouping == 'demand_segment'")
    segmented.pivot(index='group',columns='model',values='mae').reindex(columns=ORDER).plot.bar(
        ax=axes[1],color=[COLORS[m] for m in ORDER],rot=0)
    axes[1].set(title='Final MAE by demand frequency',ylabel='MAE (units)',xlabel='Segment')
    axes[1].legend(fontsize=7)
    save(fig,'segments')
    fig,ax=plt.subplots(figsize=(3.6,2.35),constrained_layout=True)
    timing=data['final']['runtime'].set_index('model').reindex(ORDER)
    position=np.arange(len(ORDER))
    ax.scatter(timing.fit_seconds,position+.1,color='#168c85',s=25,label='Fit')
    ax.scatter(timing.predict_seconds,position-.1,color='#7863ac',s=25,marker='s',label='Inference')
    ax.set_yticks(position,[NAMES[m] for m in ORDER])
    ax.set(xscale='log',xlabel='Seconds (log scale)',title='Final computational cost')
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)
    save(fig,'runtime')
    fig,ax=plt.subplots(figsize=(6.5,2.8),constrained_layout=True)
    logs=pd.read_csv(RELEASE/'final/models/origin_1913/deepar/training_history.csv')
    ax.plot(logs.epoch,logs.train_loss,label='Training NLL',color=COLORS['deepar'])
    ax.plot(logs.epoch,logs.val_loss,label='Inner validation NLL',color=COLORS['kalman'])
    ax.set(xlabel='Epoch',ylabel='Negative log likelihood',title='Global DeepAR learning curve')
    ax.legend(fontsize=8)
    save(fig,'training')
    # Machine-readable evidence for editable slide charts and tables.
    historical=pd.read_csv(ROOT/'reports/uncertainty_analysis/overall.csv')
    reference=pd.read_csv(ROOT/'reports/kalman_variant_comparison.csv').query("model == 'kalman_single_sku'").rename(columns={'MAE':'mae','RMSE':'rmse','WAPE':'wape'})
    historical=pd.concat([historical,reference],ignore_index=True)
    ablation=pd.read_csv(ROOT/'reports/deepar_experiments_comparison.csv')
    deck=dict(benchmark=metrics.replace({np.nan:None}).to_dict('records'),
              historical=historical.replace({np.nan:None}).to_dict('records'),
              ablation=ablation.replace({np.nan:None}).to_dict('records'),
              conditional=conditional.replace({np.nan:None}).to_dict('records'),
              runtime=data['final']['runtime'].to_dict('records'),
              weeks={scope:data[scope]['weeks'].replace({np.nan:None}).to_dict('records') for scope in data},
              examples=examples,training=logs.to_dict('records'),segments=segments.to_dict('records'),
              pairs=pd.DataFrame(pairs).to_dict('records'))
    (ROOT/'.build/final').mkdir(parents=True,exist_ok=True)
    (ROOT/'artifacts/presentation').mkdir(parents=True,exist_ok=True)
    json_write(ROOT/'.build/final/evidence.json',deck)
    json_write(ROOT/'artifacts/presentation/evidence.json',deck)
    # Export exactly resolved packages for the tested all-model environment.
    import importlib.metadata
    environment={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','torch','lightgbm','streamlit','matplotlib','pyarrow']}
    json_write(RELEASE/'environment.json',dict(python=sys.version,platform=platform.platform(),packages=environment,torch_threads=1,lightgbm_threads=1,kalman_workers=4))
    hashes={str(p.relative_to(RELEASE)):digest(p) for p in sorted(RELEASE.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    json_write(RELEASE/'manifest.json',dict(files=hashes,source_commit='3d1f8a8',final_evaluation_count=1))
    print('Packaged',len(hashes),'artifacts; demo includes',segments.shape[0],'series')

if __name__=='__main__':
    main()
