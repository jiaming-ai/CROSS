"""Frozen-checkpoint offline test; reports single-sample error and interventions."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .head import TrajectoryHead
from .train import load_cache,evaluate,metrics
from .backbone import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',nargs='+',required=True);p.add_argument('--cache',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--split',choices=['val','test','train'],default='test')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    results={};loaded={}
    for path in a.checkpoints:
        ck=torch.load(path,map_location='cpu',weights_only=True)
        raw=ck['train_cfg'].get('raw',False)
        if raw not in loaded: loaded[raw]=load_cache(a.cache,a.split,raw)
        data=loaded[raw]
        train_prov=ck['provenance'];backbone_ids={m['backbone_sha256'] for m in train_prov['cache_metadata']}
        assert all(m['backbone_sha256'] in backbone_ids for m in data[4]),'backbone mismatch'
        model=TrajectoryHead(**ck['head_cfg']).cuda();model.load_state_dict(ck['model']);model.eval()
        normal,pred,err=evaluate(model,data);shuffled,_,_=evaluate(model,data,shuffle=True)
        seeds=[];sample_preds=[]
        if model.mode=='flow':
            for seed in (2027,2028,2029):
                sm,sp,_=evaluate(model,data,seed=seed);seeds.append(sm);sample_preds.append(sp)
        g=data[1];straight=torch.arange(1,model.horizon+1,device='cuda')[None,:,None]*.5*g[:,None]/g.norm(dim=-1)[:,None,None]
        base,_=metrics(straight,data[2],data[3])
        tag=Path(path).parent.parent.name+'_'+Path(path).parent.name
        result={'checkpoint':path,'sha256':sha256(path),'step':ck['step'],'head_cfg':ck['head_cfg'],
                'normal':normal,'shuffled_visual':shuffled,'straight_goal':base,'extra_flow_seeds':seeds}
        if model.geometry!='none':
            result['scale_half']=evaluate(model,data,scale_factor=.5)[0]
            result['scale_double']=evaluate(model,data,scale_factor=2.)[0]
            result['odometry_scale']=evaluate(model,data,use_odometry=True)[0]
        if seeds: result['flow_mean_single_sample_ADE']=float(np.mean([normal['ADE']]+[s['ADE'] for s in seeds]))
        # Head-only batch1 latency, excludes cached feature extraction and data loading.
        xx=data[0][:1];gg=g[:1]
        for _ in range(10):
            with torch.autocast('cuda',dtype=torch.bfloat16):model.predict(xx,gg,geometry={k:v[:1] for k,v in data[5].items()})
        torch.cuda.synchronize();t=time.perf_counter()
        for _ in range(50):
            with torch.autocast('cuda',dtype=torch.bfloat16):model.predict(xx,gg,geometry={k:v[:1] for k,v in data[5].items()})
        torch.cuda.synchronize();result['head_batch1_ms']=(time.perf_counter()-t)*1000/50
        results[tag]=result
        np.savez_compressed(out/f'{tag}.npz',prediction=pred.cpu().numpy(),target=data[2].cpu().numpy(),goal=g.cpu().numpy(),errors=err,
                            ids=np.array([r['id'] for r in data[3]]),groups=np.array([r['dataset']+':'+r['group'] for r in data[3]]),
                            extra_samples=np.stack([p.cpu().numpy() for p in sample_preds]) if sample_preds else np.empty(0))
        (out/'results.json').write_text(json.dumps(results,indent=2));print(tag,json.dumps(result),flush=True)
        del model
    (out/'DONE.json').write_text(json.dumps({'checkpoints':len(results),'split':a.split},indent=2))

if __name__=='__main__':main()
