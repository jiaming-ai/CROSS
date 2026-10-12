import numpy as np
import torch
from cross_nav.policy.data import arc_window,split_group
from cross_nav.policy.head import TrajectoryHead


def test_arc_coordinates_and_causal_history():
    p=np.stack((np.zeros(201),np.arange(201)*.1),-1)
    ids,target,goal=arc_window(p,30,np.array([0,1]),np.array([-1,0]),goal_distance=6)
    assert max(ids)==30 and ids[0]==30
    np.testing.assert_allclose(target[:,0],np.arange(1,9)*.5,atol=1e-6)
    np.testing.assert_allclose(target[:,1],0,atol=1e-6)
    np.testing.assert_allclose(goal,[6,0],atol=1e-6)
    # Future changes can affect labels but must never change selected observations.
    p2=p.copy();p2[40:,0]=np.arange(161)*.05
    ids2,_,_=arc_window(p2,30,np.array([0,1]),np.array([-1,0]),goal_distance=6)
    assert ids==ids2


def test_reject_discontinuity_and_insufficient_future():
    p=np.stack((np.arange(201)*.1,np.zeros(201)),-1)
    assert arc_window(p,195,[1,0],[0,1]) is None
    p[60:]+=100
    assert arc_window(p,30,[1,0],[0,1]) is None


def test_holdout_entire_dataset():
    assert all(split_group('recon',str(i))=='test' for i in range(100))
    assert split_group('nclt','2012-02-19')==split_group('nclt','2012-02-19')


def test_head_gradients_and_flow_integration():
    torch.set_num_threads(1)
    x=torch.randn(2,4,2,2,2,16);goal=torch.randn(2,2);target=torch.randn(2,8,2)
    for source,compression,mode in [('multi',0,'regression'),('last',0,'regression'),('multi',3,'flow')]:
        m=TrajectoryHead(token_dim=16,width=24,heads=4,layers=1,source=source,compression=compression,mode=mode)
        loss=m.loss(x,goal,target);loss.backward()
        assert torch.isfinite(loss)
        assert m.proj[-1][1].weight.grad.abs().sum()>0
        assert m.goal[0].weight.grad.abs().sum()>0
        noise=torch.randn(2,8,2)
        a=m.predict(x,goal,noise=noise);b=m.predict(x,goal,noise=noise)
        torch.testing.assert_close(a,b)
        assert a.shape==target.shape and torch.isfinite(a).all()


def test_goal_only_ignores_visual_input():
    m=TrajectoryHead(token_dim=16,width=24,heads=4,layers=1,source='goal').eval()
    x=torch.randn(2,4,2,2,2,16);goal=torch.randn(2,2)
    torch.testing.assert_close(m.predict(x,goal),m.predict(x*100,goal))


def test_metric_backprojection_and_scale_inputs():
    from cross_nav.policy.backbone import anchor_points
    depth=torch.full((1,2,4,4),2.)
    E=torch.zeros(1,2,3,4);E[:,:,:3,:3]=torch.eye(3);E[:,1,0,3]=-1
    K=torch.eye(3).repeat(1,2,1,1);K[:,:,0,0]=2;K[:,:,1,1]=2;K[:,:,0,2]=1.5;K[:,:,1,2]=1.5
    xyz=anchor_points(depth,E,K,2)
    torch.testing.assert_close(xyz[:,1]-xyz[:,0],torch.tensor([1.,0,0]).expand(1,2,2,3))
    torch.testing.assert_close(xyz[...,2],torch.full((1,2,2,2),2.))
    m=TrajectoryHead(token_dim=16,width=24,heads=4,layers=1,horizon=16,geometry='metric',action_scale=8.)
    x=torch.randn(1,4,2,2,2,16);goal=torch.tensor([[10.,0.]])
    geom={'xyz_unit':xyz,'log_scale':torch.tensor([.5])}
    loss=m.loss(x,goal,torch.randn(1,16,2),geometry=geom);loss.backward()
    assert m.xyz_embed[0].weight.grad.abs().sum()>0
    assert m.scale_embed[0].weight.grad.abs().sum()>0
    assert m.predict(x,goal,geometry=geom).shape==(1,16,2)


def test_eight_metre_target():
    p=np.stack((np.arange(401)*.1,np.zeros(401)),-1)
    _,target,goal=arc_window(p,30,[1,0],[0,1],horizon=16,spacing=.5,goal_distance=10.)
    np.testing.assert_allclose(target[-1],[8,0])
    np.testing.assert_allclose(target[9],[5,0])
    np.testing.assert_allclose(goal,[10,0])


def test_small_rgb_fills_letterbox(tmp_path):
    from PIL import Image
    from cross_nav.policy.data import ImageWindows
    p=tmp_path/'rgb.jpg';Image.new('RGB',(85,64),(255,0,0)).save(p)
    images,_=ImageWindows([{'images':[str(p)]}],256)[0]
    assert images[0,0,128,0]>.9 and images[0,1,128,0]<.1


def test_flow_velocity_integration_has_correct_sign_and_units():
    import types
    m=TrajectoryHead(token_dim=16,width=24,heads=4,layers=1,mode='flow',action_scale=8.)
    def velocity(self,memory,goal,x=None,t=None):return torch.ones_like(x)*.25
    m.decode=types.MethodType(velocity,m)
    x=torch.randn(1,4,2,2,2,16);g=torch.zeros(1,2);noise=torch.zeros(1,8,2)
    torch.testing.assert_close(m.predict(x,g,steps=10,noise=noise),torch.full((1,8,2),2.))


def test_metric_shared_tokens_inference(tmp_path):
    from cross_nav.policy.infer import TrajectoryPredictor
    m=TrajectoryHead(token_dim=16,width=24,heads=4,layers=1,horizon=16,action_scale=8.,geometry='metric').eval()
    path=tmp_path/'head.pt';torch.save({'model':m.state_dict(),'head_cfg':m.cfg,'train_cfg':{}},path)
    predictor=TrajectoryPredictor(path,device='cpu')
    tokens=torch.randn(1,4,2,8,8,16);goal=torch.tensor([[10.,0.]])
    geometry={'xyz_unit':torch.randn(1,2,8,8,3),'log_scale':torch.tensor([.3])}
    with torch.autocast('cpu',dtype=torch.bfloat16):expected=m.predict(tokens,goal,geometry=geometry).float()
    torch.testing.assert_close(predictor.from_tokens(tokens,goal,geometry=geometry),expected)


def test_past_odometry_scale():
    from cross_nav.policy.infer import odometry_log_scale
    pred=torch.tensor([[0.,.5,1.,2.]])
    metric=pred*3.
    torch.testing.assert_close(odometry_log_scale(pred,metric).exp(),torch.tensor([3.]))
    import pytest
    with pytest.raises(ValueError):odometry_log_scale(pred*0,metric)
