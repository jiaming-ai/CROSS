"""Auditable causal windows and distance-indexed metric demonstration targets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zlib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def arc_window(position, anchor, forward, left, horizon=8, spacing=.5, goal_distance=6., history=(0.,.5,1.,2.)):
    p=np.asarray(position,dtype=np.float64)
    edges=np.linalg.norm(np.diff(p,axis=0),axis=-1)
    arc=np.r_[0.,np.cumsum(edges)]
    d=arc[anchor]
    if d < max(history) or arc[-1] < d+max(goal_distance,horizon*spacing): return None
    end=int(np.searchsorted(arc,d+max(goal_distance,horizon*spacing)))
    begin=int(np.searchsorted(arc,d-max(history)))
    if np.any(edges[begin:end]>1.5): return None  # pose/recording discontinuity
    basis=np.stack((forward,left),axis=1)
    local=(p-p[anchor])@basis
    s=np.arange(1,horizon+1)*spacing+d
    target=np.stack([np.interp(s,arc,local[:,i]) for i in range(2)],-1)
    goal=np.array([np.interp(d+goal_distance,arc,local[:,i]) for i in range(2)])
    # Right-sided search chooses only frames already observed at each past position.
    ids=[min(int(anchor),max(0,int(np.searchsorted(arc,d-h,side='right')-1))) for h in history]
    ids[0]=int(anchor)
    if len(set(ids))!=len(ids) or np.linalg.norm(goal)<1. or np.linalg.norm(target[-1])<.5: return None
    return ids,target.astype('float32'),goal.astype('float32')


def split_group(dataset,group):
    # Entire RECON dataset is permanently excluded from all policy fitting/tuning.
    if dataset=='recon': return 'test'
    if dataset=='cory_hall': return 'val' if any(x in group for x in ('aug10','aug13')) else 'train'
    return 'val' if zlib.crc32(f'{dataset}:{group}'.encode())%5==0 else 'train'


def scene_records(cache, bev_root, dataset):
    from vggt_ft.dataio.scene import load_scene
    for scene_path in sorted((Path(cache)/dataset).glob('*/scene.json')):
        sc=load_scene(scene_path.parent)
        if dataset=='tartanground' and sc.name in {'CarWelding','Gascola','MiddleEast','ModernCityDowntown','Supermarket','GreatMarsh'}: continue
        bp=Path(bev_root)/dataset/f'{sc.world}.npz'
        if not bp.exists(): continue
        with np.load(bp) as z: up=z['R_g'][2].astype(float)
        for name in sc.sequences:
            sq=sc.seq(name)
            if dataset=='tartanground':
                meta=json.loads(scene_path.read_text())
                if meta.get('robot',{}).get(name)!='diff': continue
            if sq.n<20: continue
            # NCLT sessions (all tiles together); other datasets use whole scenes.
            group=sc.name[:10] if dataset=='nclt' else sc.world
            forward=sq.E[:,:3,:3][:,2,:].copy()
            forward-=np.einsum('ij,j->i',forward,up)[:,None]*up[None]
            forward/=np.maximum(np.linalg.norm(forward,axis=-1,keepdims=True),1e-8)
            left=np.cross(up[None],forward)
            imgs=[str(sq.dir/'rgb'/f'{n}.jpg') for n in sq.names]
            yield dict(dataset=dataset,group=group,sequence=f'{sc.name}/{name}',position=sq.centres,
                       forward=forward,left=left,images=imgs)


def gnm_group(dataset,original):
    episode=Path(original).name
    if dataset=='cory_hall': return '_'.join(episode.split('_')[:2])
    if dataset=='sac_son': return episode.split('_')[0]
    if dataset=='recon': return episode[:17]  # robot and recording date
    raise ValueError(dataset)


def gnm_records(root,dataset):
    for p in sorted((Path(root)/dataset).glob('*/trajectory.npz')):
        with np.load(p) as z:
            pos=z['position'].copy(); yaw=z['yaw'].reshape(-1).copy()
        meta=json.loads((p.parent/'source.json').read_text())
        f=np.stack((np.cos(yaw),np.sin(yaw)),-1)
        l=np.stack((-np.sin(yaw),np.cos(yaw)),-1)
        yield dict(dataset=dataset,group=gnm_group(dataset,meta['original']),sequence=p.parent.name,position=pos,
                   forward=f,left=l,images=[str(p.parent/f'{i}.jpg') for i in range(len(pos))])


def make_manifest(records, out, counts, seed=17, horizon=8, spacing=.5):
    rng=np.random.default_rng(seed)
    candidates={k:[] for k in counts}
    rejected=0
    for rec in records:
        split=split_group(rec['dataset'],rec['group'])
        if split not in counts: continue
        # Limit each recording's contribution, sample uniformly over recordings later.
        anchors=rng.permutation(np.arange(2,len(rec['position'])-2))[:300]
        rows=[]
        for a in anchors:
            goal_distance=float(rng.uniform(horizon*spacing+.5,horizon*spacing+4.))
            v=arc_window(rec['position'],a,rec['forward'][a],rec['left'][a],goal_distance=goal_distance,horizon=horizon,spacing=spacing)
            if v is None: continue
            ids,target,goal=v
            images=[rec['images'][i] for i in ids]
            if not all(Path(p).is_file() for p in images): rejected+=1;continue
            key=f"{rec['dataset']}:{rec['sequence']}:{a}"
            rows.append(dict(id=key,dataset=rec['dataset'],group=rec['group'],sequence=rec['sequence'],
                             anchor=int(a),input_indices=ids,images=images,target=target.tolist(),goal=goal.tolist(),
                             goal_path_distance=goal_distance,history_chord=np.linalg.norm(rec['position'][ids]-rec['position'][a],axis=-1).tolist()))
        if rows: candidates[split].append(rows)
    manifest={'schema':1,'seed':seed,'coordinate':'metres forward,left','target':f'{horizon} waypoints at {spacing} m demonstrated arc length','horizon':horizon,'spacing':spacing,
              'history_m':[0,.5,1,2],'test_dataset':'recon','splits':{},'missing_images':rejected}
    for split,n in counts.items():
        groups=candidates[split]
        rows=[]
        while groups and len(rows)<n:
            order=rng.permutation(len(groups))
            for j in order:
                if groups[j] and len(rows)<n: rows.append(groups[j].pop())
            groups=[g for g in groups if g]
        manifest['splits'][split]=rows
    train_groups={(r['dataset'],r['group']) for r in manifest['splits'].get('train',[])}
    for split in ('val','test'):
        assert not train_groups & {(r['dataset'],r['group']) for r in manifest['splits'].get(split,[])}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'manifest':str(out),'counts':{k:len(v) for k,v in manifest['splits'].items()},'missing_images':rejected}),flush=True)


class ImageWindows(Dataset):
    def __init__(self,rows,size=256): self.rows,self.size=rows,size
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        images=[]
        for p in self.rows[i]['images']:
            with Image.open(p) as im:
                im=im.convert('RGB')
                # Preserve full field of view, pad to square rather than stretch/crop.
                ratio=self.size/max(im.size)
                im=im.resize((max(1,round(im.width*ratio)),max(1,round(im.height*ratio))),Image.Resampling.BILINEAR)
                canvas=Image.new('RGB',(self.size,self.size),(123,116,103))
                canvas.paste(im,((self.size-im.width)//2,(self.size-im.height)//2))
                images.append(torch.from_numpy(np.asarray(canvas).copy()).permute(2,0,1).float()/255.)
        return torch.stack(images),i


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache');p.add_argument('--bev-root');p.add_argument('--gnm-root')
    p.add_argument('--dataset',required=True);p.add_argument('--out',required=True);p.add_argument('--train',type=int,default=1024)
    p.add_argument('--val',type=int,default=128);p.add_argument('--test',type=int,default=512)
    p.add_argument('--horizon',type=int,default=8);p.add_argument('--spacing',type=float,default=.5)
    a=p.parse_args()
    records=gnm_records(a.gnm_root,a.dataset) if a.gnm_root else scene_records(a.cache,a.bev_root,a.dataset)
    make_manifest(records,a.out,{'test':a.test} if a.dataset=='recon' else {'train':a.train,'val':a.val},horizon=a.horizon,spacing=a.spacing)

if __name__=='__main__':main()
