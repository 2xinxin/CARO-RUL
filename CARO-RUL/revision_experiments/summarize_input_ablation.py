"""Aggregate run JSON by seed within device, then equal-weight devices."""
from pathlib import Path
import argparse,json,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'revision_experiments/results/A3-input-ablation'); args=p.parse_args(); out=args.output
runs=[json.loads(f.read_text()) for f in sorted(out.glob('SW*_*.json'))]
metrics=pd.DataFrame([r for run in runs for r in run['metrics']]); preds=pd.DataFrame([r for run in runs for r in run['predictions']]); keys=['target','seed','mode','evaluation']; assert not metrics.duplicated(keys).any()
assert len(metrics)==210 and len(preds)==1365
assert set(metrics['target'])=={f'SW{i}' for i in range(1,8)} and set(metrics['seed'])=={17,29,41,53,67} and set(metrics['mode'])=={'age_only','age_missing','full'} and set(metrics['evaluation'])=={'fractional_life','fixed_event_count'}
assert (metrics.parameters==14117).all()
metrics.to_csv(out/'fold_metrics.csv',index=False);preds.to_csv(out/'predictions.csv',index=False)
cols=['nmae','nrmse','mae','rmse','nbias','overestimation_rate','r2']; dev=metrics.groupby(['mode','evaluation','target'],as_index=False)[cols].mean(); dev.to_csv(out/'per_device_metrics.csv',index=False)
macro=dev.groupby(['mode','evaluation'],as_index=False)[cols].mean(); macro.to_csv(out/'macro_summary.csv',index=False)
paired=dev.pivot(index=['target','evaluation'],columns='mode',values='nmae').reset_index(); paired['full_minus_age_only']=paired['full']-paired['age_only'];paired['full_minus_age_missing']=paired['full']-paired['age_missing'];paired.to_csv(out/'paired_per_device_nmae.csv',index=False)
print(macro.to_string(index=False))
