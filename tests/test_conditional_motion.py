import numpy as np
import pytest

torch=pytest.importorskip('torch')
pp=pytest.importorskip('pypose')

from cross.core.conditional import SourceFactor,SourceState
from cross.core.conditional_pose import exp,inverse,log
from cross.core.odom_accum import OdomAccumulator


def test_accumulated_response_tracks_multiple_consumer_resets_and_shared_sources():
    accum=OdomAccumulator(device='cpu')
    accum.register_item('frame');accum.register_item('node')
    rng=np.random.default_rng(988)
    increments=[];factors=[]
    for i in range(6):
        T=exp(rng.normal(size=6)*.1)
        J=np.r_[-T[:3,:3].T@T[:3,3],np.zeros(3)][:,None]
        f=SourceFactor((f'image{i%2}',),J,np.array([.0144]),log_depth_scale=True)
        accum.update_odom(T,np.eye(6)*1e-4,f)
        increments.append(T);factors.append(f)
        if i==2: accum.reset_item('frame')
    for name,start in [('frame',3),('node',0)]:
        factor=accum.source_since_last_reading(name)
        nominal,_=accum.get_since_last_reading(name,reset=False)
        for c,key in enumerate(factor.keys):
            poses=[]
            for bias in (-1e-5,1e-5):
                T=np.eye(4)
                for pose,f in zip(increments[start:],factors[start:]):
                    pose=pose.copy()
                    if key in f.keys: pose[:3,3]*=np.exp(-bias)
                    T=T@pose
                poses.append(T)
            response=(log(inverse(nominal.matrix().double().numpy())@poses[1])-
                      log(inverse(nominal.matrix().double().numpy())@poses[0]))/2e-5
            np.testing.assert_allclose(factor.jacobian[:,c],response,atol=2e-7)
    accum.reset_item('node')
    assert accum.source_since_last_reading('node').keys==()


def test_raw_scale_factor_uses_exact_positive_scale_at_nonzero_bias():
    f=SourceFactor(('image',),np.array([[-2.],[0.],[0.],[0.],[0.],[0.]]),np.array([.0144]),log_depth_scale=True)
    state,_,_=SourceState(np.eye(6)).expand(f)
    state.mean[:]=np.log(1.25)
    _,J,offset=state.expand(f)
    assert 2.+offset[0]==pytest.approx(1.6)
    assert J[0,0]==pytest.approx(-1.6)
