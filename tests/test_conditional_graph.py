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


def test_repeated_graph_solves_do_not_amplify_near_unit_stored_quaternions():
    hm = chain()
    frame = lie(exp(np.array([.3,-.1,.2,0.,2.05,0.])))
    for node in hm.nodes.values():
        node.pose_mu[0] = frame @ node.pose_mu[0]
        # Magnitude observed before the fifth-session graph failure.
        node.pose_mu[0,3:] *= 1.+1.5e-6
    hm.dist = (hm.nodes[10].pose_mu.clone(), hm.dist[1], hm.dist[2])
    expected = [node.pose_mu[0].tensor().double().numpy().copy() for node in hm.nodes.values()]
    for value in expected:
        value[3:] /= np.linalg.norm(value[3:])
    for _ in range(12):
        pg = solve(hm)
        result = hm.apply_pgo_result(dict(pose_graph=pg,optimized_poses=pg.optimized_poses,other_hypothesis_id=0))
        assert result['success']
        for node,mean in zip(hm.nodes.values(),expected):
            value = node.pose_mu[0].tensor().double().numpy()
            assert abs(np.linalg.norm(value[3:])-1) < 2e-7
            np.testing.assert_allclose(value[:3],mean[:3],atol=3e-5)
            assert abs(value[3:]@mean[3:]) == pytest.approx(1.,abs=3e-7)


def test_invalid_unsolved_pose_cannot_partially_publish_graph_results():
    hm = chain()
    hm.source_states[0].mean[:] = .07
    extra = copy.deepcopy(hm.nodes[10]); extra.id = 20
    extra.pose_mu[0,3:] *= 1.2
    hm.nodes[20] = extra  # Same chart, outside the connected optimized graph.
    pg = solve(hm)
    assert 20 not in pg.optimized_poses
    before = [(n.pose_mu.clone(),n.pose_std.clone(),n.pose_charts.clone(),n.last_pgo_step,
               [m.record() if m is not None else None for m in n.conditional_poses]) for n in hm.nodes.values()]
    tracking = tuple(x.clone() for x in hm.dist)
    with pytest.raises(ValueError):
        hm.apply_pgo_result(dict(pose_graph=pg,optimized_poses=pg.optimized_poses,other_hypothesis_id=0))
    for node,snapshot in zip(hm.nodes.values(),before):
        for actual,expected in zip((node.pose_mu,node.pose_std,node.pose_charts),snapshot[:3]):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert node.last_pgo_step == snapshot[3]
        assert [m.record() if m is not None else None for m in node.conditional_poses] == snapshot[4]
    for actual,expected in zip(hm.dist,tracking):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


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


def test_saved_nonrepresentative_edges_introduce_one_unused_prior_during_pgo():
    hm = chain()
    hm.source_states[0].mean[:] = .07
    hm.source_states[0].covariance[:] = .004
    # Both raw edges passed geometry but neither was the online cluster
    # representative. Their shared metric prediction is not in the live state.
    for target in (4,8):
        pose = hm.nodes[0].pose_mu[0].Inv()@hm.nodes[target].pose_mu[0]
        T = pose.matrix().double().numpy()
        J = np.r_[-T[:3,:3].T@T[:3,3],np.zeros(3)][:,None]
        edge = ConditionalPose(np.eye(6)*.0025,SourceFactor(('unused/image',),J,np.array([.0144]),
                               factor_id=f'raw-{target}',log_depth_scale=True))
        hm.add_edge(0,target,pose,pp.se3(torch.full((6,),.05)),EdgeType.VISUAL,conditional_pose=edge)
    record = copy.deepcopy(hm.save_state())
    assert record['source_belief']['keys']==['session-A/image']
    restored = manager()
    restored.load_state(record,SimpleNamespace(),'cpu','cpu',{})
    restored.initialize_source_filter()
    pg = solve(restored)
    assert pg.conditional_belief.keys==('session-A/image','unused/image')
    np.testing.assert_array_equal(pg.conditional_belief.mean,[.07,0.])
    np.testing.assert_array_equal(pg.conditional_belief.covariance,np.diag([.004,.0144]))
    # Preparing/solving a graph has not mutated the saved tracking posterior.
    assert restored.source_states[0].keys==('session-A/image',)


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


@pytest.mark.parametrize('schmidt',[False,True])
def test_commitment_joins_charts_without_identifying_two_sessions_biases(schmidt):
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
    if schmidt:
        hm.schmidt_map_geometry=True
        hm.nodes[0].conditional_poses[0].geometry_covariance[:]=0
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
    np.testing.assert_allclose(hm.nodes[2].conditional_poses[0].factor.jacobian[0,:2],[-4.5,3.],atol=2e-5)
    assert hm.source_states[0].keys[:2]==keys
    np.testing.assert_array_equal(hm.source_states[0].covariance[:2,:2],before)
    if schmidt:
        assert len(hm.source_states[0].keys)==68
        assert hm.nodes[2].conditional_poses[0].geometry_covariance.max()==0
    else:
        assert hm.source_states[0].keys==keys
    assert hm.source_states[1] is None and 1 not in hm.hypotheses


@pytest.mark.parametrize('conditional',[False,True])
def test_initial_map_node_does_not_share_mutable_tracking_pose(conditional):
    from cross.core.system import System
    from cross.core.odom_accum import OdomAccumulator
    class Database:
        get_all_atlases=lambda self: []
        get_size=lambda self: 0
        create_atlas=lambda self: None
        def insert(self,index,rgb,depth,**kw):
            return Keyframe(kw['mu'],kw['sigma'],kw['weights'],pose_charts=kw['pose_charts'])
    system=System.__new__(System)
    system.config=SystemConfig();system.config.mapping.hypothesis.conditional_sources=conditional
    system.device=system.storage_device='cpu';system.topo_map=None
    system.kf_gmm_n_components=2;system._processed_frame_num=1;system.db=Database()
    system.hypothesis_manager=HypothesisManager(system,2,HypothesisConfig(chart_aware=True,session_recovery=True))
    system.odom_accumulator=OdomAccumulator(device='cpu')
    node=system._init_system(torch.zeros(3,8,8),None)
    before=tuple(x.clone() for x in (node.pose_mu,node.pose_std,node.pose_weights))
    delta=pp.identity_SE3();delta[0]=1.
    source=SourceFactor(('image',),np.array([[-1.],[0.],[0.],[0.],[0.],[0.]]),np.array([.0144]),log_depth_scale=True)
    system.hypothesis_manager.motion_update(delta,pp.se3(torch.full((6,),.01)),
                                            source_factor=source if conditional else None)
    for actual,snapshot in zip((node.pose_mu,node.pose_std,node.pose_weights),before):
        torch.testing.assert_close(actual,snapshot,rtol=0,atol=0)
    assert float(system.hypothesis_manager.dist[0][0,0])==1.


def schmidt_chain():
    hm=chain()
    hm.schmidt_map_geometry=True
    hm.source_states[1]=None
    hm.nodes[0].conditional_poses[0].geometry_covariance[:]=0
    return hm


def apply_graph(hm,pg):
    return hm.apply_pgo_result(dict(pose_graph=pg,optimized_poses=pg.optimized_poses,other_hypothesis_id=0))


def test_joint_graph_geometry_matches_propagating_independent_odometry_errors():
    from cross.core.conditional_pgo import _matrix
    hm=schmidt_chain();pg=solve(hm)
    # A tree's joint pose covariance has an independent construction by
    # propagating each raw increment noise through the later increments.
    responses=[]
    for i in range(1,11):
        row=np.zeros((6,60))
        for j in range(1,i+1):
            after=inverse(_matrix(pg.optimized_poses[j]))@_matrix(pg.optimized_poses[i])
            row[:,6*(j-1):6*j]=adjoint(inverse(after))
        responses.append(row)
    H=np.vstack(responses)
    Q=np.kron(np.eye(10),hm.odom_edges[0,1].conditional_pose.geometry_covariance)
    np.testing.assert_allclose(pg.joint_geometry['covariance'],H@Q@H.T,rtol=1e-5,atol=1e-9)
    assert apply_graph(hm,pg)['success']
    belief=hm.source_states[0]
    assert len(belief.keys)==61
    assert all(np.count_nonzero(n.conditional_poses[0].geometry_covariance)==0 for n in hm.nodes.values())
    a,b=hm.nodes[9].conditional_poses[0].factor.jacobian,hm.nodes[10].conditional_poses[0].factor.jacobian
    # Neighbor uncertainty is mostly shared, not two independent pose errors.
    difference=a-b
    shared=np.trace(difference[:,1:]@belief.covariance[1:,1:]@difference[:,1:].T)
    independent=np.trace(a[:,1:]@belief.covariance[1:,1:]@a[:,1:].T+b[:,1:]@belief.covariance[1:,1:]@b[:,1:].T)
    assert shared < independent*.1


def test_schmidt_refresh_marginalizes_old_geometry_and_map_roundtrip_keeps_correlations():
    from cross.core.schmidt import schmidt_product
    hm=schmidt_chain();assert apply_graph(hm,solve(hm))['success']
    before=hm.source_states[0]
    A=hm.nodes[3].conditional_poses[0].factor.jacobian
    factor=SourceFactor(before.keys,A,before.prior_variances,before.mean,'new-retrieval')
    updated=schmidt_product(before,np.zeros(6),np.eye(6)*.001,factor,
                            frozen_keys=before.keys[1:]).state
    assert np.linalg.norm(updated.covariance[0,1:]) > 1e-6
    hm.source_states[0]=updated
    record=copy.deepcopy(hm.save_state())
    restored=manager();restored.schmidt_map_geometry=True
    restored.load_state(record,SimpleNamespace(),'cpu','cpu',{})
    restored.initialize_source_filter()
    np.testing.assert_array_equal(restored.source_states[0].covariance,updated.covariance)
    for node in restored.nodes.values():
        np.testing.assert_array_equal(node.conditional_poses[0].factor.jacobian,
                                      hm.nodes[node.id].conditional_poses[0].factor.jacobian)
    old_keys=set(updated.keys[1:]);metric_variance=updated.covariance[:1,:1].copy()
    assert apply_graph(hm,solve(hm))['success']
    final=hm.source_states[0]
    assert not old_keys.intersection(final.keys)
    assert len(final.keys)==len(updated.keys)
    np.testing.assert_array_equal(final.covariance[:1,:1],metric_variance)
    np.testing.assert_array_equal(final.mean[:1],updated.mean[:1])
    assert final.seen_factors==updated.seen_factors
    np.testing.assert_array_equal(final.covariance[:1,1:],np.zeros((1,60)))


def test_schmidt_refresh_rejects_live_displacement_before_publishing_any_state():
    hm=schmidt_chain()
    hm.dist[0][0,0]+=.03
    pg=solve(hm)
    before=copy.deepcopy(hm.save_state())
    source=hm.source_states[0].record()
    poses=[n.pose_mu.clone() for n in hm.nodes.values()]
    with pytest.raises(ValueError,match='current pose to be a graph node'):
        apply_graph(hm,pg)
    assert hm.source_states[0].record()==source
    assert hm.save_state()['source_belief']==before['source_belief']
    for node,pose in zip(hm.nodes.values(),poses):
        torch.testing.assert_close(node.pose_mu,pose,atol=0,rtol=0)
        assert not any(k.startswith('geometry:') for k in node.conditional_poses[0].factor.keys)


def test_schmidt_refresh_refuses_discarding_another_mode_or_a_stochastic_gauge():
    hm=schmidt_chain();hm.source_states[1]=hm.source_states[0].copy()
    with pytest.raises(ValueError,match='one surviving mode'):
        solve(hm)
    hm.source_states[1]=None;hm.nodes[0].conditional_poses[0].geometry_covariance=np.eye(6)*.01
    with pytest.raises(ValueError,match='deterministic gauge'):
        solve(hm)
