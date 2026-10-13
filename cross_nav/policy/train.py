"""Bounded, dataset-balanced head ablations with validation-only selection."""
from __future__ import annotations
import argparse,json,time,math,random
from pathlib import Path
import numpy as np
import torch
from .head import TrajectoryHead
from .backbone import sha256


def load_cache(roots,split,raw=False,device='cuda'):
    files=[];metas=[]
    for root in roots:
        root=Path(root)
        if not (root/'DONE.json').exists(): raise RuntimeError(f'cache not ready: {root}')
        metas.append(json.loads((root/'meta.json').read_text()))
        files.extend(sorted((root/split).glob('*.pt')))
    if not files: raise ValueError(f'empty {split}')
    key='last_raw' if raw else 'tokens'
    first=torch.load(files[0],map_location='cpu',weights_only=True,mmap=True)
    X=torch.empty((len(files),*first[key].shape),dtype=first[key].dtype,device=device)
    goals=[];targets=[];rows=[];geometry=[]
    for i,p in enumerate(files):
        d=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        X[i].copy_(d[key]);geometry.append({k:d[k] for k in ('xyz_unit','log_scale','pred_history_chord','history_chord') if k in d});goals.append(d['goal']);targets.append(d['target'])
        rows.append({k:d[k] for k in ('id','dataset','group')})
    assert len({r['id'] for r in rows})==len(rows),'duplicate examples'
    if split!='test': assert all(r['dataset']!='recon' for r in rows),'held-out dataset leaked'
    geom={k:torch.stack([d[k] for d in geometry]).to(device) for k in geometry[0]}
    return X,torch.stack(goals).to(device),torch.stack(targets).to(device),rows,metas,geom


def metrics(pred,target,rows):
    err=(pred-target).norm(dim=-1).float().cpu().numpy()
    result={'n':len(rows),'ADE':float(err.mean()),'FDE':float(err[:,-1].mean()),'E5m':float(err[:,9].mean()) if err.shape[1]>=10 else None,'by_dataset':{}}
    for ds in sorted({r['dataset'] for r in rows}):
        idx=[i for i,r in enumerate(rows) if r['dataset']==ds]
        result['by_dataset'][ds]={'n':len(idx),'ADE':float(err[idx].mean()),'FDE':float(err[idx,-1].mean())}
    result['macro_ADE']=float(np.mean([x['ADE'] for x in result['by_dataset'].values()]))
    return result,err


@torch.no_grad()
def evaluate(model,data,batch=32,shuffle=False,seed=2026,scale_factor=1.,use_odometry=False):
    X,g,y,rows,_,geom=data;model.eval();pred=[]
    generator=torch.Generator(device=X.device).manual_seed(seed)
    geom={k:v.clone() for k,v in geom.items()}
    if 'log_scale' in geom:
        if use_odometry:
            from .infer import odometry_log_scale
            geom['log_scale']=odometry_log_scale(geom['pred_history_chord'],geom['history_chord'])
        geom['log_scale']+=math.log(scale_factor)
    noise=torch.randn(len(rows),model.horizon,2,device=X.device,generator=generator)
    perm=torch.arange(len(rows),device=X.device)
    if shuffle:
        # Preserve dataset distribution; break observation-target pairing.
        for ds in sorted({r['dataset'] for r in rows}):
            idx=torch.tensor([i for i,r in enumerate(rows) if r['dataset']==ds],device=X.device)
            perm[idx]=idx[torch.randperm(len(idx),device=X.device,generator=generator)]
    for i in range(0,len(rows),batch):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred.append(model.predict(X[perm[i:i+batch]],g[i:i+batch],noise=noise[i:i+batch],geometry={k:v[perm[i:i+batch]] for k,v in geom.items()}).float())
    pred=torch.cat(pred)
    result,err=metrics(pred,y,rows)
    return result,pred,err


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--source',choices=['last','multi','goal'],default='multi');p.add_argument('--mode',choices=['regression','flow'],default='regression')
    p.add_argument('--compression',type=int,default=0);p.add_argument('--raw',action='store_true')
    p.add_argument('--steps',type=int,default=1500);p.add_argument('--batch',type=int,default=32);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--init')
    p.add_argument('--geometry',choices=['none','scalar','metric'],default='none');p.add_argument('--horizon',type=int,default=8)
    p.add_argument('--spacing',type=float,default=.5)
    p.add_argument('--scale-conditioning',choices=['mono','mixed'],default='mono');a=p.parse_args()
    if a.raw and a.source!='last': raise ValueError('raw cache contains only last layer')
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cfg=vars(a);(out/'config.json').write_text(json.dumps(cfg,indent=2))
    train=load_cache(a.cache,'train',a.raw);val=load_cache(a.cache,'val',a.raw)
    train_groups={(r['dataset'],r['group']) for r in train[3]}
    assert not train_groups & {(r['dataset'],r['group']) for r in val[3]},'split group leakage'
    model=TrajectoryHead(source=a.source,mode=a.mode,compression=a.compression,geometry=a.geometry,horizon=a.horizon,action_scale=a.horizon*a.spacing).cuda()
    if a.init:model.load_state_dict(torch.load(a.init,map_location='cpu',weights_only=True)['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    X,g,y,rows,metas,geom=train
    assert y.shape[1]==a.horizon
    if a.geometry!="none" and not geom:raise ValueError("metric cache required")
    sets=sorted({r['dataset'] for r in rows})
    ids=[torch.tensor([i for i,r in enumerate(rows) if r['dataset']==s],device='cuda') for s in sets]
    receipt={'datasets':{s:sum(r['dataset']==s for r in rows) for s in sets},'sampling_weight':{s:1/len(sets) for s in sets},
             'cache_metadata':metas,'validation_cache_metadata':val[4],'parameters':sum(p.numel() for p in model.parameters()),'initialization':a.init,'effective_batch':math.ceil(a.batch/len(sets))*len(sets)}
    (out/'provenance.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True);best=float('inf');t0=time.time();log=[]
    for step in range(a.steps+1):
        if step%250==0 or step==a.steps:
            result,_,_=evaluate(model,val)
            result.update(step=step,seconds=time.time()-t0);log.append(result)
            print(json.dumps(result),flush=True)
            (out/'validation.json').write_text(json.dumps(log,indent=2))
            checkpoint={'model':model.state_dict(),'head_cfg':model.cfg,'step':step,'validation':result,'train_cfg':cfg,'provenance':receipt}
            if result['macro_ADE']<best:
                best=result['macro_ADE'];torch.save(checkpoint,out/'best.pt')
            if step==a.steps:torch.save(checkpoint,out/'final.pt');break
        model.train()
        idx=torch.cat([v[torch.randint(len(v),(math.ceil(a.batch/len(ids)),),device='cuda')] for v in ids])
        idx=idx[torch.randperm(len(idx),device='cuda')]
        lr=a.lr*.5*(1+math.cos(math.pi*step/a.steps))
        for pg in opt.param_groups:pg['lr']=lr
        opt.zero_grad(set_to_none=True)
        batch_geom={k:v[idx] for k,v in geom.items()}
        if a.scale_conditioning=='mixed':
            if a.geometry=='none':raise ValueError('mixed scale requires scale-aware head')
            from .infer import odometry_log_scale
            odom=odometry_log_scale(batch_geom['pred_history_chord'],batch_geom['history_chord'])
            use_odom=torch.rand(len(idx),device='cuda')<.5
            batch_geom['log_scale']=torch.where(use_odom,odom,batch_geom['log_scale'])+.1*torch.randn_like(odom)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(X[idx],g[idx],y[idx],geometry=batch_geom)
        if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss at {step}')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if (step+1)%100==0:print(json.dumps({'step':step+1,'loss':float(loss),'seconds':time.time()-t0}),flush=True)
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model'])
    normal,pred,err=evaluate(model,val);shuffled,_,_=evaluate(model,val,shuffle=True)
    straight=g.new_tensor(np.arange(1,a.horizon+1)*a.spacing)[None,:,None]*val[1][:,None]/val[1].norm(dim=-1)[:,None,None]
    base,_=metrics(straight,val[2],val[3])
    summary={'best_step':ck['step'],'normal':normal,'shuffled_visual':shuffled,'straight_goal':base,'checkpoint_sha256':sha256(out/'best.pt')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(out/'val_predictions.npz',prediction=pred.cpu().numpy(),target=val[2].cpu().numpy(),goal=val[1].cpu().numpy(),errors=err,
                        ids=np.array([r['id'] for r in val[3]]),groups=np.array([r['dataset']+':'+r['group'] for r in val[3]]))
    (out/'DONE.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
