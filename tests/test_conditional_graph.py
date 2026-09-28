"""Exercise actual GTSAM solves, map persistence and global-message selection."""
from types import SimpleNamespace
import copy

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pp = pytest.importorskip('pypose')
pytest.importorskip('gtsam')

from cross.core.config import HypothesisConfig, SystemConfig
from cross.core.conditional import SourceFactor,SourceState
from cross.core.conditional_pose import ConditionalPose,adjoint,exp,inverse,log,compose
from cross.core.hypothesis import HypothesisManager
from cross.core.pgo import PoseGraph
from cross.core.types import Keyframe,EdgeType


def manager(components=2):
    system = SimpleNamespace(device='cpu',topo_map=None,config=SystemConfig(),last_added_kf_id=None)
    hm = HypothesisManager(system,components,HypothesisConfig(chart_aware=True,session_recovery=True))
    hm.dist = (pp.identity_SE3(components),pp.se3(torch.full((components,6),.01)),
               torch.tensor([1.]+[0.]*(components-1)))
    hm.start_tracking_chart(); hm.initialize_source_filter()
    return hm


def lie(T):
    return pp.from_matrix(torch.as_tensor(T,dtype=torch.float32),pp.SE3_type)


def chain():
    hm = manager()
    J = np.zeros((6,1)); J[0,0] = -.25
    relative = exp(np.array([.25,0.,.02,0.,.04,.01]))
    # Response need not be a pure axis-aligned scale, to exercise SE3 transport.
    J[:3,0] = -relative[:3,:3].T@relative[:3,3]
    factor = SourceFactor(('session-A/image',),J,np.array([.0144]),log_depth_scale=True)
    edge = ConditionalPose(np.diag([.01**2]*3+[.02**2]*3),factor)
    pose = np.eye(4)
    for i in range(11):
        if i:
            hm.motion_update(lie(relative),pp.se3(torch.tensor([.01]*3+[.02]*3)),source_factor=factor)
        mu,std,weights = hm.get_active_dist()
        node = Keyframe(mu.clone(),std.clone(),weights.clone(),pose_charts=hm.get_active_charts(),temporary=True)
        node.id = i
        hm.add_node(node)
        if i:
            hm.add_edge(i-1,i,lie(relative),pp.se3(torch.tensor([.01]*3+[.02]*3)),EdgeType.ODOMETRY,
                        conditional_pose=edge)
    hm.system.last_added_kf_id = 10
    return hm


def solve(hm):
    pg = PoseGraph(hm,device='cpu')
    pg.construct_for_loop_closure(10,0)
    pg.solve(set(range(1,11)),{0})
    return pg


def test_graph_response_matches_resolving_at_perturbed_bias_and_keeps_prior_once():
    hm = chain()
    nominal = solve(hm)
    J = nominal.optimized_conditional_poses[10].factor.jacobian.copy()
    before = hm.source_states[0].covariance.copy()
    responses = []
    for b in (-1e-3,1e-3):
        hm.source_states[0].mean[:] = b
        pg = solve(hm)
        responses.append(pg.optimized_poses[10].matrix().double().numpy())
    numerical = (log(inverse(nominal.optimized_poses[10].matrix().double().numpy())@responses[1])-
                 log(inverse(nominal.optimized_poses[10].matrix().double().numpy())@responses[0]))/.002
    np.testing.assert_allclose(J[:,0],numerical,atol=2e-4)
    hm.source_states[0].mean[:] = 0.
    result = hm.apply_pgo_result(dict(pose_graph=nominal,optimized_poses=nominal.optimized_poses,other_hypothesis_id=0))
    assert result['success']
    np.testing.assert_array_equal(hm.source_states[0].covariance,before)
    np.testing.assert_allclose(hm.source_states[0].jacobian,J,atol=1e-10)
    assert nominal.conditional_response_diagnostics['bias_updated'] is False
    assert hm.nodes[10].conditional_poses[0].factor.keys == ('session-A/image',)


def test_stale_conditional_graph_is_rejected_before_mutating_any_node():
    hm = chain(); pg = solve(hm)
    before = [n.pose_mu.clone() for n in hm.nodes.values()]
    hm.dist[0][0,0] += .01
    result = hm.apply_pgo_result(dict(pose_graph=pg,optimized_poses=pg.optimized_poses,other_hypothesis_id=0))
    assert not result['success'] and 'stale' in result['message']
    for node,pose in zip(hm.nodes.values(),before):
        torch.testing.assert_close(node.pose_mu,pose,rtol=0,atol=0)


def test_saved_node_conditional_model_is_separate_from_one_bias_posterior():
    hm = chain()
    hm.source_states[0].mean[:] = .07
    hm.source_states[0].covariance[:] = .004
    record = copy.deepcopy(hm.save_state())
    assert all('covariance' not in k['conditional_poses'][0]['factor'] for k in record['temp_keyframes'])
    restored = manager()
    restored.load_state(record,SimpleNamespace(), 'cpu','cpu',{})
    restored.initialize_source_filter()
    for state in restored.source_states:
        np.testing.assert_array_equal(state.mean,[.07])
        np.testing.assert_array_equal(state.covariance,[[.004]])
        np.testing.assert_array_equal(state.jacobian,np.zeros((6,1)))
    assert restored.nodes[10].conditional_poses[0].factor.keys == ('session-A/image',)
    assert restored.odom_edges[9,10].conditional_pose.factor.log_depth_scale


def test_representative_message_preserves_full_covariance_and_actual_source():
    from cross.core.system import System
    hm = manager()
    system = System.__new__(System)
    system.device,system.config,system.hypothesis_manager = 'cpu',SystemConfig(),hm
    frames=[]; models=[]
    for i,score in enumerate((.6,.9)):
        node = Keyframe(pp.identity_SE3(2),pp.se3(torch.full((2,6),.1)),torch.tensor([1.,0.]),
                        pose_charts=torch.tensor([0,-1]))
        node.id=i+100; node.pose_mu[0,0]=.03*i
        J=np.zeros((6,1));J[0,0]=i+1.
        R=np.eye(6)*.01;R[0,1]=R[1,0]=.003
        model=ConditionalPose(R,SourceFactor((str(i),),J,np.array([.0144]),factor_id=str(i)))
        frames.append(node);models.append([model,None])
    ret=dict(convolved_mus=torch.stack([n.pose_mu for n in frames]),convolved_stds=torch.stack([n.pose_std for n in frames]),
             pose_est_conf=torch.tensor([.6,.9]),valid_ref_component_weights=torch.stack([n.pose_weights for n in frames]),
             valid_keyframes=frames,convolved_conditional_poses=models)
    mu,std,weight,confidence,edges=system._merge_and_align_components(ret)
    assert hm.aligned_conditional_models[0].factor.factor_id=='1'
    assert hm.aligned_conditional_models[0].geometry_covariance[0,1]==.003
    hm.gmm_filtering(mu,std,weight,confidence,
        source_factors={i:m.factor for i,m in hm.aligned_conditional_models.items()},
        source_covariances={i:m.geometry_covariance for i,m in hm.aligned_conditional_models.items()})
    assert hm.source_states[0].keys==('1',)


def test_commitment_joins_charts_without_identifying_two_sessions_biases():
    from cross.core.hypothesis import Hypothesis
    hm=manager()
    hm.component_charts[:]=torch.tensor([1,0])
    hm.realized[:]=True;hm.ttl[:]=10
    hm.hypotheses[1]=Hypothesis(1,5)
    keys=('reference/image','query/image')
    belief,_,_=SourceState(np.eye(6)*.01).expand(SourceFactor(keys,np.zeros((6,2)),np.array([.0144,.0144])))
    def pose(x):
        value=pp.identity_SE3();value[0]=x;return value
    def model(j1,j2):
        J=np.zeros((6,2));J[0]=[j1,j2]
        return ConditionalPose(np.eye(6)*.01,SourceFactor(keys,J,np.array([.0144,.0144])))
    for i in range(12):
        mu=pp.identity_SE3(2);std=pp.se3(torch.full((2,6),.1));weight=torch.tensor([1.,0.])
        mu[0]=pose(i if i<2 else 20+i-2)
        models=[model(-i,0) if i<2 else model(0,-(i-2)),None]
        charts=torch.tensor([0 if i<2 else 1,-1])
        if i>=5:
            mu[1]=pose(4.5+i-5);models[1]=model(-4.5,-(i-5));charts[1]=0;weight[:]=.5
        node=Keyframe(mu,std,weight,pose_charts=charts,conditional_poses=models,temporary=True)
        node.id=i;hm.nodes[i]=node
    hm.dist=(hm.nodes[11].pose_mu.clone(),hm.nodes[11].pose_std.clone(),torch.tensor([.5,.5]))
    hm.source_states=[belief.with_pose(m)[0] for m in hm.nodes[11].conditional_poses]
    for a,b,j1,j2 in [(0,1,-1.,0.)]+[(i,i+1,0.,-1.) for i in range(2,11)]:
        hm.add_edge(a,b,pose(1.),pp.se3(torch.full((6,),.1)),EdgeType.ODOMETRY,conditional_pose=model(j1,j2))
    hm.add_edge(1,5,pose(3.5),pp.se3(torch.full((6,),.1)),EdgeType.VISUAL,
                to_comp_id=1,conditional_pose=model(-3.5,0.))
    hm.system.last_added_kf_id=11
    pg=PoseGraph(hm,device='cpu')
    pg.construct_for_loop_closure(11,1)
    assert len(pg.vertices)==12  # physical nodes, no duplicate odometry copies
    assert sum(len(f) for a,b,f in pg.edges)==11
    pg.solve(set(range(1,12)),{0})
    np.testing.assert_allclose(pg.optimized_conditional_poses[11].factor.jacobian[0],[-4.5,-6.],atol=2e-5)
    before=hm.source_states[1].covariance.copy()
    result=hm.apply_pgo_result(dict(pose_graph=pg,optimized_poses=pg.optimized_poses,other_hypothesis_id=1))
    assert result['success']
    assert all(int(node.pose_charts[0])==0 for node in hm.nodes.values())
    assert hm.dist[0][0,0]==pytest.approx(10.5,abs=1e-5)
    assert hm.nodes[2].pose_mu[0,0]==pytest.approx(1.5,abs=1e-5)
    np.testing.assert_allclose(hm.nodes[2].conditional_poses[0].factor.jacobian[0],[-4.5,3.],atol=2e-5)
    assert hm.source_states[0].keys==keys
    np.testing.assert_array_equal(hm.source_states[0].covariance,before)
    assert hm.source_states[1] is None and 1 not in hm.hypotheses
