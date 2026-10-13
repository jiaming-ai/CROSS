"""Convert checksum-verified Open-X GNM TFRecords to RGB/metric pose episodes.

Only numeric arrays and encoded images are read; no remote pickle execution.
Source recording prefixes, not random frames, define validation groups.
"""
import argparse,base64,hashlib,io,json
from pathlib import Path
import numpy as np
from PIL import Image
from tfrecord.reader import tfrecord_loader
from .data import gnm_group


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',required=True);p.add_argument('--dataset',required=True)
    a=p.parse_args();raw=Path(a.raw);out=Path(a.out)/a.dataset;out.mkdir(parents=True,exist_ok=True)
    objects=json.loads((raw/'objects.json').read_text())
    source='berkeley_gnm_'+a.dataset
    objects=[x for x in objects if f'/{source}/' in x['name']]
    episodes=0;frames=0;groups=set();lengths=[];first=[]
    for obj in objects:
        path=raw/source/Path(obj['name']).name
        if not path.exists():raise FileNotFoundError(path)
        assert base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()==obj['md5Hash']
        for record in tfrecord_loader(str(path),None):
            original=record['episode_metadata/file_path'].decode()
            episode=Path(original).name
            # GNM sequence segments share a recording prefix, e.g. cory5_aug17_00_0546.
            group=gnm_group(a.dataset,original)
            pos=np.asarray(record['steps/observation/position']).reshape(-1,2).astype('float64')
            yaw=np.asarray(record['steps/observation/yaw']).reshape(-1).astype('float64')
            images=record['steps/observation/image']
            if isinstance(images,bytes):images=[images]
            assert len(pos)==len(yaw)==len(images)
            assert np.isfinite(pos).all() and np.isfinite(yaw).all()
            key=hashlib.sha256(original.encode()).hexdigest()[:20]
            dest=out/key
            if (dest/'source.json').exists():
                old=json.loads((dest/'source.json').read_text())
                if old['original']!=original:raise ValueError('episode hash collision')
                continue
            dest.mkdir(exist_ok=True)
            for i,im in enumerate(images):
                with Image.open(io.BytesIO(im)) as img:
                    img.convert('RGB').save(dest/f'{i}.jpg',quality=95)
            np.savez_compressed(dest/'trajectory.npz',position=pos,yaw=yaw)
            (dest/'source.json').write_text(json.dumps({'original':original,'group':group,'source_object':obj,'metric':True},indent=2))
            frames+=len(pos);episodes+=1;groups.add(group)
            lengths.append(float(np.linalg.norm(np.diff(pos,axis=0),axis=-1).sum()))
            if len(first)<10:first.append(original)
        print(a.dataset,path.name,episodes,frames,flush=True)
    receipt={'new_episodes':episodes,'new_frames':frames,'groups':sorted(groups),'examples':first,
             'arc_length_quantiles':np.quantile(lengths,[0,.25,.5,.75,1]).tolist() if lengths else [],'source_objects':objects}
    (out/'READY.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='source_objects'}),flush=True)

if __name__=='__main__':main()
