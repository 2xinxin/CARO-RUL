"""Rebuild final manuscript diagnostic CSVs from retained historical predictions."""
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'generated'/'paper_results'
OUT.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'results'/'PCSS_RICH_BENCHMARK'
p=pd.read_csv(BASE/'B016_CARO_UNIFIED_5SEED'/'predictions.csv')
p=p[p.model.eq('caro_rul')].copy()
p['signed_error']=p.predicted_rul-p.true_rul
p['event_mae']=p.signed_error.abs()
p.groupby(['evaluation','target'])[['event_mae','signed_error']].mean().to_csv(OUT/'B016_event_errors_by_device.csv')
p=pd.read_csv(ROOT/'revision_experiments/results/A2-002/predictions.csv')
methods=['iorh_q375','iorh_fixed_width_010','tcn_source_constant']
p=p[(p.evaluation=='online_256')&p.method.isin(methods)].copy()
rows=[]
for (method,target,seed),g in p.groupby(['method','target','seed']):
    g=g.sort_values('index'); h=g[g.predicted_rul<=512]; selected=h.iloc[0] if len(h) else g.iloc[-1]; y=float(selected.true_rul)
    assert g.iloc[-1].true_rul==1
    rows.append(dict(method=method,target=target,seed=seed,premature_events=max(y-512,0),late_events=max(512-y,0),missed=int(h.empty),hit_at_last=int(not h.empty and selected['index']==g.iloc[-1]['index']),selected_true_rul=y,selected_index=int(selected['index'])))
inter=pd.DataFrame(rows); assert len(inter)==105
for key,op in [('early',inter.selected_true_rul>512),('late',inter.selected_true_rul<512),('exact',inter.selected_true_rul==512)]:
    inter[key]=((inter.missed==0)&op).astype(int)
inter.to_csv(OUT/'intervention_outcomes_by_run.csv',index=False)
g=p[(p.method==methods[1])&(p['index']%256==0)].copy(); g['nmae']=(g.predicted_rul-g.true_rul).abs()/g.lifetime
at=g.groupby(['index','target']).nmae.mean().reset_index().groupby('index').agg(eligible_devices=('target','nunique'),nmae=('nmae','mean')).reset_index(); assert len(at)==78
at.to_csv(OUT/'at_risk_checkpoints.csv',index=False)
from sys import path
path.insert(0,str(ROOT/'revision_experiments'))
import analyze_revision_evidence as ev
# Use the original quantile diagnostic implementation via its CLI to regenerate the complete diagnostics.
print(inter.groupby('method')[['early','late','exact','missed','hit_at_last']].sum())
print('Wrote',OUT)
