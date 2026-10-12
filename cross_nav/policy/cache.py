"""Materialize frozen tokens once so controlled head ablations share inputs."""
import argparse,json,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .backbone import FrozenTokens,FrozenMetricTokens,sha256
from .data import ImageWindows


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);p.add_argument('--batch',type=int,default=4);p.add_argument('--size',type=int,default=256)
    p.add_argument('--metric',action='store_true')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    m=json.loads(Path(a.manifest).read_text())
    meta={'manifest_sha256':sha256(a.manifest),'backbone_sha256':sha256(a.checkpoint),'size':a.size,'pool':8,'layers':[4,11,17,23]}
    if a.metric:meta.update(metric=True,preprocessing='letterbox_resize_up_or_down',horizon=m.get('horizon',8),spacing=m.get('spacing',.5))
    if (out/'meta.json').exists() and json.loads((out/'meta.json').read_text())!=meta: raise ValueError('cache provenance mismatch')
    (out/'meta.json').write_text(json.dumps(meta,indent=2))
    model=(FrozenMetricTokens if a.metric else FrozenTokens)(a.checkpoint).cuda().eval();t0=time.time()
    for split,rows in m['splits'].items():
        dest=out/split;dest.mkdir(exist_ok=True)
        todo=[(i,r) for i,r in enumerate(rows) if not (dest/f'{i:06d}.pt').exists()]
        loader=DataLoader(ImageWindows([r for i,r in todo],a.size),batch_size=a.batch,num_workers=4,pin_memory=True)
        for imgs,indices in loader:
            features=model(imgs.cuda(non_blocking=True),last_raw=True)
            if a.metric:pooled,raw=features['tokens'],features['last_raw']
            else:pooled,raw=features
            for j,idx in enumerate(indices.tolist()):
                i,r=todo[idx]
                path=dest/f'{i:06d}.pt'
                payload={'tokens':pooled[j].cpu(),'last_raw':raw[j].cpu(),'goal':torch.tensor(r['goal']),
                            'target':torch.tensor(r['target']),'id':r['id'],'dataset':r['dataset'],'group':r['group']}
                if a.metric:
                    payload.update({k:features[k][j].cpu() for k in ('xyz_unit','log_scale','pred_history_chord')})
                    payload['history_chord']=torch.tensor(r['history_chord'])
                torch.save(payload,path.with_suffix('.tmp'))
                path.with_suffix('.tmp').rename(path)
            if int(indices[-1])%64<a.batch: print(split,int(indices[-1])+1,len(todo),'seconds',round(time.time()-t0),flush=True)
        assert len(list(dest.glob('*.pt')))==len(rows)
    (out/'DONE.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':main()
