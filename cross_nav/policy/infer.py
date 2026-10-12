"""Standalone or shared-backbone policy inference in metres (forward, left)."""
import torch
from torch.nn import functional as F
from .head import TrajectoryHead
from .backbone import FrozenTokens,FrozenMetricTokens


class TrajectoryPredictor:
    def __init__(self,head_checkpoint,backbone_checkpoint=None,device='cuda'):
        self.device=torch.device(device)
        ck=torch.load(head_checkpoint,map_location='cpu',weights_only=True)
        self.head=TrajectoryHead(**ck['head_cfg']).to(device).eval()
        self.head.load_state_dict(ck['model'])
        self.raw=ck['train_cfg'].get('raw',False)
        self.image_size=ck.get('provenance',{}).get('cache_metadata',[{'size':256}])[0]['size']
        self.backbone=(FrozenMetricTokens if self.head.geometry!="none" else FrozenTokens)(backbone_checkpoint).to(device).eval() if backbone_checkpoint else None

    @torch.no_grad()
    def from_tokens(self,tokens,goal,noise=None,geometry=None):
        """tokens B,L,S,H,W,C in training layer/view order; goal B,2 metres."""
        with torch.autocast(self.device.type,dtype=torch.bfloat16):
            return self.head.predict(tokens.to(self.device),goal.to(self.device),steps=10,
                                     noise=None if noise is None else noise.to(self.device),
                                     geometry=None if geometry is None else {k:v.to(self.device) for k,v in geometry.items()}).float()

    @torch.no_grad()
    def from_aggregator(self,tokens_list,patch_token_start,patch_hw,goal,noise=None,geometry=None):
        """Reuse an existing CROSS pass; frames must be current-first and causal.

        Select the same current / .5 / 1 / 2 m history used in training. Different
        view counts or resolution are architectural possibilities, not validated
        operating points for the supplied experiment checkpoints.
        """
        layers=(23,) if self.raw else (4,11,17,23)
        grids=[];h,w=patch_hw
        for l in layers:
            t=tokens_list[l][:,:,patch_token_start:]
            B,S,_,C=t.shape
            t=t.reshape(B*S,h,w,C).permute(0,3,1,2)
            if not self.raw:t=F.adaptive_avg_pool2d(t,(8,8))
            hh,ww=t.shape[-2:]
            grids.append(t.permute(0,2,3,1).reshape(B,S,hh,ww,C))
        return self.from_tokens(torch.stack(grids,1),goal,noise,geometry)

    @torch.no_grad()
    def __call__(self,images,goal,noise=None,log_scale=None):
        """images B,4,3,H,W float [0,1], at the checkpoint resolution, letterboxed; goal B,2 metric."""
        if self.backbone is None:raise RuntimeError('provide backbone checkpoint or use from_aggregator')
        if images.shape[1:]!=(4,3,self.image_size,self.image_size):raise ValueError(f'checkpoint expects four {self.image_size}px views')
        features=self.backbone(images.to(self.device),last_raw=True)
        if self.head.geometry!="none":
            if log_scale is not None:features['log_scale']=log_scale.to(self.device)
            return self.from_tokens(features['last_raw'] if self.raw else features['tokens'],goal,noise,
                                    {k:features[k] for k in ('xyz_unit','log_scale')})
        pooled,raw=features
        return self.from_tokens(raw if self.raw else pooled,goal,noise)

    @torch.no_grad()
    def from_backbone(self,bb,goal,log_scale=None,noise=None):
        """Reuse CROSS/NavModel outputs. log_scale may come from stereo/odometry.

        Geometry remains in camera0 OpenCV axes; the returned path uses
        forward/left in the robot's horizontal navigation frame.
        """
        from .backbone import anchor_points
        geometry=None
        if self.head.geometry!='none':
            if log_scale is None:log_scale=bb['log_scale_head']
            geometry={'log_scale':log_scale,'xyz_unit':anchor_points(bb['depth'],bb['extrinsics'],bb['intrinsics'],8)}
        return self.from_aggregator(bb['tokens_list'],bb['pts'],bb['patch_hw'],goal,noise,geometry)


def odometry_log_scale(predicted_chords,metric_chords):
    """Robust scale from current-to-past camera-centre distances only.

    Both tensors B,S; exclude current zero distance and near-zero predicted
    baselines. Camera/body lever arms must already be accounted for by caller.
    """
    valid=(predicted_chords>1e-3)&(metric_chords>1e-3)
    ratios=torch.where(valid,metric_chords/predicted_chords.clamp_min(1e-3),torch.nan)
    scale=ratios.nanmedian(dim=1).values
    if not torch.isfinite(scale).all() or not (scale>0).all():raise ValueError('no valid metric scale anchor')
    return scale.log()
