"""Capacity-matched input ablation; original B016 trainer and preprocessing."""
import argparse, json, sys, time
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np, torch, yaml
from pcss_rich_benchmark.data import load_records, fit_transform, transformed_record, build_samples, make_sample
from pcss_rich_benchmark.run_baselines import metrics
from tor_rul.dash_experiment import fit_neural, predict_direct_log_rul

def restrict(sample, mode):
    if mode=='full': return sample
    seq=sample.sequence.copy(); seq[:42 if mode=='age_only' else 28]=0
    tab=sample.tabular.copy(); tab[:70 if mode=='age_only' else 56]=0
    assert seq.shape==(43,128) and np.array_equal(seq[-1],sample.sequence[-1])
    return replace(sample,sequence=seq,tabular=tab)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--target',nargs='+'); p.add_argument('--seed',nargs='+',type=int); p.add_argument('--mode',nargs='+',choices=['age_only','age_missing','full'],default=['age_only','age_missing']); p.add_argument('--threads',type=int,default=2); p.add_argument('--device',default='cpu'); p.add_argument('--output',type=Path,default=ROOT/'revision_experiments/results/A3-input-ablation'); args=p.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    device=torch.device(args.device); cfg=yaml.safe_load((ROOT/'configs/b016_caro_unified_5seed.yaml').read_text()); protocol=cfg['protocol']; training=cfg['training']; records=load_records(ROOT/cfg['dataset']['root'],cfg['dataset']['records']); out=args.output; out.mkdir(parents=True,exist_ok=True)
    for target in args.target or list(records):
        rec=records[target]; src=[r for k,r in records.items() if k!=target]; tf=fit_transform(src,cfg['dataset']['feature_columns'],protocol['preprocessing_fit_max_fraction']); tr={k:transformed_record(r,tf) for k,r in records.items()}; w=protocol['window']
        train=[s for r in src for s in build_samples(r,tr[r.record_id],w,*protocol['train_fraction'],protocol['train_samples_per_source'])]; val=[s for r in src for s in build_samples(r,tr[r.record_id],w,*protocol['validation_fraction'],protocol['validation_samples_per_source'])]
        ev={'fractional_life':[make_sample(rec,tr[target],int(np.floor(x*(rec.length-1))),w) for x in protocol['test_checkpoints']], 'fixed_event_count':[make_sample(rec,tr[target],x,w) for x in protocol['fixed_event_indices'] if w<=x<rec.length-1]}
        for mode in args.mode:
            train_m=[restrict(s,mode) for s in train]; val_m=[restrict(s,mode) for s in val]; ev_m={k:[restrict(s,mode) for s in v] for k,v in ev.items()}
            for seed in args.seed or training['screening_seeds']:
                dest=out/f'{target}_{seed}_{mode}.json'
                if dest.exists(): continue
                start=time.monotonic(); fit=fit_neural('caro_rul',train_m,val_m,window=w,hidden=training['hidden_size'],bins=training['hazard_bins'],epochs=training['epochs'],patience=training['patience'],batch_size=training['batch_size'],learning_rate=training['learning_rate'],weight_decay=training['weight_decay'],gradient_clip=training['gradient_clip'],seed=seed,device=device,target_mode='log_pulses'); assert fit.parameters==14117
                rows=[]; preds=[]
                for evaluation,samples in ev_m.items():
                    prediction,_=predict_direct_log_rul(fit,samples,device)
                    rows.append(dict(target=target,seed=seed,mode=mode,evaluation=evaluation,parameters=fit.parameters,epochs=fit.epochs,validation_mae=fit.best_validation,**metrics(samples,prediction)))
                    preds.extend(dict(target=target,seed=seed,mode=mode,evaluation=evaluation,index=s.index,true_rul=s.target_pulses,predicted_rul=float(y),lifetime=s.lifetime) for s,y in zip(samples,prediction))
                payload=dict(metrics=rows,predictions=preds,torch_version=str(torch.__version__),device=str(device),threads=args.threads,seconds=time.monotonic()-start)
                temp=dest.with_suffix('.tmp'); temp.write_text(json.dumps(payload,indent=2)); temp.replace(dest)
                print(target,seed,mode,[(r['evaluation'],round(r['nmae'],5)) for r in rows],round(payload['seconds'],2),flush=True)
if __name__=='__main__': main()
