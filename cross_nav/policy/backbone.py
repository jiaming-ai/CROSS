"""Frozen aggregator only: no dense geometry head or ground truth at inference."""
from pathlib import Path
import hashlib
import torch
from torch import nn
from torch.nn import functional as F


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()


class FrozenTokens(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        from vggt_omega.models.aggregator import Aggregator
        self.aggregator = Aggregator()
        sd=torch.load(checkpoint,map_location='cpu',weights_only=False)
        sd=sd.get('model',sd)
        sd={k[len('aggregator.'):]:v for k,v in sd.items() if k.startswith('aggregator.')}
        self.aggregator.load_state_dict(sd,strict=True)
        self.aggregator.requires_grad_(False).eval().to(torch.bfloat16)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images, pool=8, last_raw=False):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            layers,pts=self.aggregator(images.to(torch.bfloat16))
        B,S,_,H,W=images.shape
        h,w=H//16,W//16
        def grid(t,size):
            t=t[:,:,pts:].reshape(B*S,h,w,-1).permute(0,3,1,2)
            if size and (h,w)!=(size,size): t=F.adaptive_avg_pool2d(t,(size,size))
            hh,ww=t.shape[-2:]
            return t.permute(0,2,3,1).reshape(B,S,hh,ww,-1)
        pooled=torch.stack([grid(layers[i],pool) for i in (4,11,17,23)],1)
        return (pooled,grid(layers[23],0)[:,None]) if last_raw else pooled


def anchor_points(depth, extrinsics, intrinsics, grid=8):
    """Backproject depth cell means into camera0 in VGGT internal units.

    Camera-from-world E: p_w=R_s.T(p_s-t_s); p_0=R_0 p_w+t_0.
    Pixel coordinates below match adaptive average pooling cell centres.
    """
    B,S,H,W=depth.shape
    d=F.adaptive_avg_pool2d(depth.reshape(B*S,1,H,W),(grid,grid)).reshape(B,S,grid,grid)
    yy,xx=torch.meshgrid((torch.arange(grid,device=depth.device)+.5)*H/grid-.5,
                         (torch.arange(grid,device=depth.device)+.5)*W/grid-.5,indexing='ij')
    K=intrinsics
    xyz=torch.stack(((xx-K[:,:,0,2,None,None])/K[:,:,0,0,None,None]*d,
                     (yy-K[:,:,1,2,None,None])/K[:,:,1,1,None,None]*d,d),-1)
    R,t=extrinsics[:,:,:3,:3],extrinsics[:,:,:3,3]
    world=torch.einsum('bsij,bshwi->bshwj',R,xyz-t[:,:,None,None])
    return torch.einsum('bij,bshwj->bshwi',R[:,0],world)+t[:,0,None,None,None]


class FrozenMetricTokens(FrozenTokens):
    """VGGT tokens, camera/depth and the shipped dense scale head, all frozen."""
    def __init__(self,checkpoint):
        nn.Module.__init__(self)
        from vggt_ft.model import VGGTOmegaFT
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model=VGGTOmegaFT(scale_head=True,covis_head=False,scale_head_cfg=ck['scale_head_cfg'])
        missing,unexpected=self.model.load_state_dict(ck['model'],strict=False)
        bad=[k for k in missing if not k.startswith(('covis_head.','scale_encoder.'))]
        if bad:raise RuntimeError(bad)
        self.model.requires_grad_(False).eval()
        self.model.aggregator.to(torch.bfloat16)

    @torch.no_grad()
    def forward(self,images,pool=8,last_raw=False):
        from vggt_omega.utils.pose_enc import encoding_to_camera
        m=self.model;agg=m.aggregator;B,S,_,H,W=images.shape;h,w=H//16,W//16
        dino={}
        hook=agg.patch_embed.register_forward_hook(lambda module,args,out:dino.__setitem__('x',out['x_norm_patchtokens'] if isinstance(out,dict) else out))
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16):layers,pts=agg(images.to(torch.bfloat16))
        finally:hook.remove()
        def grid(t,size):
            t=t[:,:,pts:].reshape(B*S,h,w,-1).permute(0,3,1,2)
            if size:t=F.adaptive_avg_pool2d(t,(size,size))
            hh,ww=t.shape[-2:]
            return t.permute(0,2,3,1).reshape(B,S,hh,ww,-1)
        tokens=torch.stack([grid(layers[i],pool) for i in (4,11,17,23)],1)
        raw=grid(layers[-1],0)[:,None]
        with torch.autocast('cuda',enabled=False):
            tl=[t.float() if t is not None else None for t in layers]
            pe=m.camera_head(tl,patch_token_start=pts)
            depth,conf=m.dense_head(tl,images=images.float(),patch_token_start=pts)
            E,K=encoding_to_camera(pe,(H,W))
            focal=.5/torch.tan(pe[...,8].float()/2)
            scale=m.scale_head(tl,pts,(h,w),dino['x'].reshape(B,S,h*w,-1).float(),focal)
            log_scale=scale[0] if isinstance(scale,tuple) else scale
            xyz=anchor_points(depth[...,0],E,K,pool)
            centres=-torch.einsum('bsji,bsj->bsi',E[:,:,:3,:3],E[:,:,:3,3])
            chord=(centres-centres[:,:1]).norm(dim=-1)
        return {'tokens':tokens,'last_raw':raw,'xyz_unit':xyz,'log_scale':log_scale,'pred_history_chord':chord}
