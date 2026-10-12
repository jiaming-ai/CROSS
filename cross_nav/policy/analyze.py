"""Paired errors with recording-cluster bootstrap and curved-route slices."""
import argparse,json
from pathlib import Path
import numpy as np


def cluster_interval(values,groups,seed=17,n_boot=2000):
    groups=np.asarray(groups);unique=np.unique(groups)
    sums=np.array([values[groups==g].sum() for g in unique]);counts=np.array([(groups==g).sum() for g in unique])
    rng=np.random.default_rng(seed);draw=rng.integers(len(unique),size=(n_boot,len(unique)))
    estimates=sums[draw].sum(1)/counts[draw].sum(1)
    return {'mean':float(values.mean()),'ci95':np.quantile(estimates,[.025,.975]).tolist(),'groups':len(unique)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result={};loaded={}
    for path in a.predictions:
        z=np.load(path);pred=z['prediction'];target=z['target'];goal=z['goal'];h=target.shape[1]
        straight=np.arange(1,h+1)[None,:,None]*.5*goal[:,None]/np.linalg.norm(goal,axis=-1)[:,None,None]
        straight_err=np.linalg.norm(straight-target,axis=-1).mean(1)
        err=np.linalg.norm(pred-target,axis=-1)
        curved=straight_err>.5
        p_len=np.linalg.norm(np.diff(np.concatenate([np.zeros_like(pred[:,:1]),pred],1),axis=1),axis=-1).sum(1)
        t_len=np.linalg.norm(np.diff(np.concatenate([np.zeros_like(target[:,:1]),target],1),axis=1),axis=-1).sum(1)
        name=Path(path).stem
        if name=='val_predictions':name=Path(path).parent.name
        loaded[name]=z
        result[name]={'ADE':cluster_interval(err.mean(1),z['groups']),'FDE':float(err[:,-1].mean()),
                      'curve_n':int(curved.sum()),'curve_ADE':float(err[curved].mean()) if curved.any() else None,
                      'straight_curve_ADE':float(straight_err[curved].mean()) if curved.any() else None,
                      'path_length_ratio_median':float(np.median(p_len/t_len)),
                      'abs_log_length_ratio':float(np.abs(np.log(np.maximum(p_len,1e-6)/t_len)).mean())}
    result['paired']={}
    names=list(loaded)
    for i,n in enumerate(names):
        for m in names[i+1:]:
            x,y=loaded[n],loaded[m]
            if not np.array_equal(x['ids'],y['ids']):continue
            dx=np.linalg.norm(x['prediction']-x['target'],axis=-1).mean(1)
            dy=np.linalg.norm(y['prediction']-y['target'],axis=-1).mean(1)
            result['paired'][n+' minus '+m]=cluster_interval(dx-dy,x['groups'])
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
