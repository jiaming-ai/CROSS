"""Use the actual CROSS motion/filter/lifecycle with conditional source states."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pp = pytest.importorskip('pypose')

from cross.core.conditional import SourceFactor,SourceState
from cross.core.hypothesis import HypothesisManager


def manager(n=1, std=.1):
    value = HypothesisManager(SimpleNamespace(device='cpu',topo_map=None),n)
    value.dist = (pp.identity_SE3(n),pp.se3(torch.full((n,6),std)),torch.ones(n)/n)
    value.ttl[:] = 10
    return value


def factor(frame, variance=.09, j=1.):
    J = np.zeros((6,1)); J[0,0] = j
    return SourceFactor(('teacher-image',),J,np.array([variance]),factor_id=f'pair-{frame}')


def test_zero_bias_variance_at_common_mean_recovers_inherited_update_and_evidence():
    old,new = manager(2),manager(2)
    new.initialize_source_filter()
    for frame in range(5):
        observation = pp.identity_SE3(2)
        std,weights,confidence = pp.se3(torch.full((2,6),.08+.01*frame)),torch.tensor([.6,.4]),torch.tensor([.8,.6])
        old.gmm_filtering(observation,std,weights,confidence)
        new.gmm_filtering(observation,std,weights,confidence,
                          source_factors={i:factor(frame,variance=0.) for i in range(2)})
        for a,b in zip(old.dist,new.dist):
            torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        for name in ('llr_hist','log_c_hist','log_conf_hist','last_sum_pos','last_hit_rate'):
            torch.testing.assert_close(getattr(old,name),getattr(new,name),rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(old.ttl,new.ttl)


def test_shared_motion_error_accumulates_by_source_identity():
    hm = manager(std=0.)
    hm.initialize_source_filter()
    motion = pp.identity_SE3(); motion[0] = 1.
    f = factor('motion',j=-1.)
    for _ in range(3):
        hm.motion_update(motion,pp.se3(torch.zeros(6)),source_factor=f)
    assert hm.dist[0][0,0] == pytest.approx(3.)
    assert hm.dist[1][0,0] == pytest.approx(.9)
    np.testing.assert_array_equal(hm.source_states[0].jacobian[:,0],[-3.,0.,0.,0.,0.,0.])
    assert hm.source_states[0].covariance[0,0] == .09
    with pytest.raises(ValueError,match='motion factor'):
        hm.motion_update(motion,pp.se3(torch.zeros(6)))


def test_duplicate_delivery_changes_neither_pose_bias_nor_commitment_evidence():
    hm = manager()
    hm.initialize_source_filter()
    observation = pp.identity_SE3(1); observation[0,0] = .3
    args = (observation,pp.se3(torch.full((1,6),.05)),torch.ones(1),torch.ones(1))
    f = factor(1)
    hm.gmm_filtering(*args,source_factors={0:f})
    before = tuple(x.clone() for x in hm.dist)
    state = hm.source_states[0].copy()
    history,ttl,pointer = hm.llr_hist.clone(),hm.ttl.clone(),hm.llr_hist_ptr
    observation[0,0] = 20.  # same factor ID cannot manufacture new information
    for _ in range(8):
        hm.gmm_filtering(*args,source_factors={0:f})
    for actual,expected in zip(hm.dist,before):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    np.testing.assert_array_equal(hm.source_states[0].mean,state.mean)
    np.testing.assert_array_equal(hm.source_states[0].covariance,state.covariance)
    torch.testing.assert_close(hm.llr_hist,history)
    torch.testing.assert_close(hm.ttl,ttl)
    assert hm.llr_hist_ptr == pointer


def test_pose_gating_also_preserves_bias_and_cross_covariance():
    hm = manager()
    hm.initialize_source_filter()
    state,_,_ = hm.source_states[0].expand(factor(0))
    state.jacobian[0,0] = -2.
    hm.source_states[0] = state
    hm.dist[1][0] = pp.se3(torch.as_tensor(state.marginal_covariance().diagonal().copy(),dtype=torch.float32).sqrt())
    pose,std = hm.dist[0].clone(),hm.dist[1].clone()
    observation = pp.identity_SE3(1); observation[0,0] = .6
    hm.gmm_filtering(observation,pp.se3(torch.full((1,6),.02)),torch.ones(1),torch.ones(1),
        pose_update_mask=torch.tensor([False]),source_factors={0:factor(1)})
    torch.testing.assert_close(hm.dist[0],pose)
    torch.testing.assert_close(hm.dist[1],std)
    np.testing.assert_array_equal(hm.source_states[0].mean,state.mean)
    np.testing.assert_array_equal(hm.source_states[0].covariance,state.covariance)
    np.testing.assert_array_equal(hm.source_states[0].jacobian,state.jacobian)
    assert 'pair-1' in hm.source_states[0].seen_factors


def test_source_correlation_affects_actual_mixture_overlap_and_weights():
    joint,independent = manager(2,std=.02),manager(2,std=.02)
    joint.initialize_source_filter()
    for component in range(2):
        state,_,_ = joint.source_states[component].expand(factor(0))
        state.jacobian[0,0] = 1.
        joint.source_states[component] = state
        marginal = torch.as_tensor(state.marginal_covariance().diagonal().copy(),dtype=torch.float32).sqrt()
        joint.dist[1][component] = pp.se3(marginal)
        independent.dist[1][component] = pp.se3(marginal)
    proposal = pp.identity_SE3(2); proposal[1,0] = .5
    conditional_std = pp.se3(torch.full((2,6),.02))
    independent_std = conditional_std.clone()
    independent_std[:,0] = (.02**2+.09)**.5
    independent.gmm_filtering(proposal,independent_std,torch.ones(2)/2,torch.ones(2))
    joint.gmm_filtering(proposal,conditional_std,torch.ones(2)/2,torch.ones(2),
        source_factors={0:factor(1),1:factor(1)})
    # Equal teacher sensitivities cancel from innovation; their uncertainty
    # cannot explain a contradictory displacement in the same-source factor.
    assert joint.dist[2][1] < independent.dist[2][1]*1e-4
    assert joint.log_c_hist[1,0] < independent.log_c_hist[1,0]-20.
    assert joint.source_states[0].covariance[0,0] == .09


def test_recycled_slot_clears_conditional_state_and_newborn_seeds_one_bias_prior():
    hm = manager(2)
    hm.initialize_source_filter()
    observation = pp.identity_SE3(2); observation[0,0] = .2
    hm.gmm_filtering(observation,pp.se3(torch.full((2,6),.03)),torch.ones(2)/2,torch.ones(2),
        source_factors={0:factor(1),1:factor(1)})
    bias_before = hm.source_states[0].copy()
    hm._reset_component_evidence(1)
    assert hm.source_states[1] is None
    hm.newborn[1],hm.ttl[1] = True,10
    observation[1,0] = 5.
    hm.gmm_filtering(observation,pp.se3(torch.full((2,6),.1)),torch.ones(2)/2,torch.ones(2),
        source_factors={1:factor(2,j=2.)})
    np.testing.assert_array_equal(hm.source_states[1].covariance,bias_before.covariance)
    np.testing.assert_array_equal(hm.source_states[1].mean,bias_before.mean)
    assert hm.dist[0][1,0] == pytest.approx(5.+2*bias_before.mean[0],abs=1e-6)
    assert hm.last_sum_pos[1] == 0


def test_incomplete_graph_adapter_cannot_silently_drop_conditional_information():
    hm = manager(); hm.initialize_source_filter()
    with pytest.raises(ValueError,match='source responses'):
        hm.apply_pgo_result({})
    assert hm.save_state()['source_belief']['version'] == 1
    with pytest.raises(NotImplementedError,match='merging'):
        hm.merge_hypotheses(1)
    with pytest.raises(NotImplementedError,match='promoting'):
        hm.change_hypo_to_first(1)


def test_invalid_motion_source_prior_cannot_partially_advance_the_mixture():
    hm = manager(2); hm.initialize_source_filter()
    hm.source_states[0],_,_ = hm.source_states[0].expand(factor(0,variance=.09))
    hm.source_states[1],_,_ = hm.source_states[1].expand(factor(0,variance=.04))
    before = tuple(x.clone() for x in hm.dist)
    states = [s.copy() for s in hm.source_states]
    motion = pp.identity_SE3(); motion[0] = 1.
    with pytest.raises(ValueError,match='different prior'):
        hm.motion_update(motion,pp.se3(torch.zeros(6)),source_factor=factor(1))
    for actual,expected in zip(hm.dist,before):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for actual,expected in zip(hm.source_states,states):
        np.testing.assert_array_equal(actual.geometry_covariance,expected.geometry_covariance)
        np.testing.assert_array_equal(actual.jacobian,expected.jacobian)


def test_repeated_se3_mean_updates_do_not_amplify_quaternion_roundoff():
    hm=manager();hm.initialize_source_filter()
    # Near-unit input of the magnitude seen in the failed image run.
    hm.dist[0][0,3:] *= 1.+8e-7
    increment=pp.se3(torch.tensor([.02,.01,0.,.03,-.02,.04])).Exp()
    for i in range(200):
        proposal=hm.dist[0]@increment
        hm.gmm_filtering(proposal,pp.se3(torch.full((1,6),.08)),torch.ones(1),torch.ones(1),
                         source_factors={0:SourceFactor((),np.empty((6,0)),np.empty(0),factor_id=str(i))})
        q=hm.dist[0][0].tensor()[3:]
        assert abs(float(q.double().norm())-1)<5e-7
