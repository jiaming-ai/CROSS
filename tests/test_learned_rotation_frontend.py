"""Exercise camera direction, anchor renewal and metric-only translation."""
from types import MethodType, SimpleNamespace

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cross.mono.config import MonoConfig, ScaleConfig
from cross.mono.geometry import inverse
from cross.mono.pnp_frontend import LearnedRotationPnPFrontend
from cross.mono.refinement import XFeatRefiner
from cross.mono.scale import LogScaleFilter


def fixture(depth_scale=1., invalid_prediction=False):
    rng=np.random.default_rng(12)
    K=np.array([[520.,0.,320.],[0.,520.,240.],[0.,0.,1.]])
    pixels=np.array([(x,y) for y in range(80,401,40) for x in range(100,541,40)],float)
    z=rng.uniform(2.,5.,len(pixels))
    points=np.c_[pixels,np.ones(len(pixels))] @ np.linalg.inv(K).T * z[:,None]
    poses=[np.eye(4) for _ in range(3)]
    poses[1][:3,:3]=cv2.Rodrigues(np.array([.025,-.045,.015]))[0]
    poses[2][:3,:3]=poses[1][:3,:3] @ cv2.Rodrigues(np.array([-.015,.035,.025]))[0]
    poses[1][:3,3]=[.08,.015,.03];poses[2][:3,3]=[.15,-.025,.01]
    features=[];depths=[]
    for i,pose in enumerate(poses):
        camera=points @ pose[:3,:3]-pose[:3,3] @ pose[:3,:3]
        uvz=camera @ K.T;uv=uvz[:,:2]/uvz[:,2:]
        depth=np.full((480,640),np.nan)
        xy=np.rint(uv).astype(int);depth[xy[:,1],xy[:,0]]=camera[:,2]*depth_scale
        features.append(dict(keypoints=torch.tensor(uv),shape=(480,640),frame=i))
        depths.append(depth)
    refiner=XFeatRefiner.__new__(XFeatRefiner)
    refiner.K=K;refiner.subpixel=refiner.rotation_selection=False;refiner.matcher="mnn"
    refiner.last_correspondences=0;refiner.last_rotation_only=False
    refiner.extract=lambda rgb:features[int(rgb[0,0,0])]
    refiner.match=MethodType(lambda self,a,b:(a['keypoints'].numpy(),b['keypoints'].numpy()),refiner)
    frontend=LearnedRotationPnPFrontend.__new__(LearnedRotationPnPFrontend)
    frontend.config=MonoConfig(frontend='learned_rotation_pnp',scale=ScaleConfig(interval=1))
    frontend.K=K;frontend.refiner=refiner;frontend.index=frontend.anchor_index=0
    frontend.anchor_features=frontend.anchor_depth=None
    frontend.metric_pose=frontend.anchor_pose=np.eye(4)
    frontend.last_timestamp=None;frontend.provide_mapping_depth=False
    frontend.rotation_prior=frontend.rotation_anchor_image=None
    frontend.scale_filter=LogScaleFilter(frontend.config.scale)
    frontend.scale_filter.initialized=True
    frontend.metric=SimpleNamespace(predict_metric=lambda rgb,K,shape:depths[int(rgb[0,0,0])])
    pairs=[]
    def predict(images):
        pairs.append(tuple(images))
        # A non-identity model world gauge and unrelated translation must not
        # affect the learned relative rotation or supply metric translation.
        gauge=np.eye(4);gauge[:3,:3]=cv2.Rodrigues(np.array([.2,-.1,.3]))[0]
        ext=np.array([inverse(poses[i]) @ gauge for i in images])
        ext[:,:3,3]=np.array([[100.,20.,-30.],[-200.,35.,60.]])
        if invalid_prediction=='nonfinite_rotation':ext[1,:3,:3]=np.nan
        elif invalid_prediction=='implausible_rotation':
            ext[1,:3,:3]=cv2.Rodrigues(np.array([2.,0.,0.]))[0] @ ext[0,:3,:3]
        return SimpleNamespace(extrinsics=ext)
    frontend.geometry=SimpleNamespace(prepare=lambda rgb:int(rgb[0,0,0]),predict=predict)
    return frontend,poses,pairs


@pytest.mark.parametrize('depth_scale',[1.,2.])
def test_learned_rotation_renews_with_metric_anchor_and_translation_uses_only_metric_depth(depth_scale):
    frontend,poses,pairs=fixture(depth_scale)
    outputs=[]
    for i in range(3):
        estimate=frontend.step(np.full((2,2,3),i,np.uint8),float(i))
        expected=poses[i].copy();expected[:3,3]*=depth_scale
        np.testing.assert_allclose(estimate.pose,expected,atol=2e-5)
        assert estimate.diagnostics['valid']
        assert estimate.diagnostics['learned_rotation_used']==(i>0)
        outputs.append(estimate.pose.copy())
    assert pairs==[(0,1),(1,2)]
    np.testing.assert_array_equal(outputs[0],np.eye(4))


@pytest.mark.parametrize('reason',['nonfinite_rotation','implausible_rotation'])
def test_invalid_prediction_falls_back_to_geometric_pnp_without_poisoning_state(reason):
    frontend,poses,pairs=fixture(invalid_prediction=reason)
    frontend.step(np.zeros((2,2,3),np.uint8),0.)
    result=frontend.step(np.ones((2,2,3),np.uint8),1.)
    assert result.diagnostics['valid'] and not result.diagnostics['learned_rotation_used']
    assert result.diagnostics['learned_rotation_reason']==reason
    np.testing.assert_allclose(result.pose,poses[1],atol=2e-5)
    with pytest.raises(ValueError,match='timestamps'):
        frontend.step(np.ones((2,2,3),np.uint8),1.)
    assert len(pairs)==1
