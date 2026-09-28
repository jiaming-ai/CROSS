"""Independent batch conditioning, source identity and correlation limits."""
import numpy as np
import pytest

from cross.core.conditional import SourceFactor, SourceState, conditional_product


@pytest.mark.parametrize('log_depth_scale', [False, True])
def test_interleaved_source_extension_keeps_joint_belief_and_exact_response(log_depth_scale):
    state = SourceState(np.eye(6), ('a','b','c'), np.array([.2,-.1,.3]),
        np.array([[.04,.01,.02],[.01,.03,-.01],[.02,-.01,.06]]),
        np.arange(18).reshape(6,3)/20, np.array([.09,.16,.25]))
    old = state.copy()
    J = np.arange(30).reshape(6,5)/30
    if log_depth_scale:
        J[3:] = 0
    factor = SourceFactor(('c','new1','a','new2','b'), J,
        np.array([.25,.36,.09,.49,.16]), np.array([.1,.2,-.1,-.2,.3]),
        log_depth_scale=log_depth_scale)
    extended, response, offset = state.expand(factor)
    assert extended.keys == ('a','b','c','new1','new2')
    np.testing.assert_array_equal(extended.mean, [.2,-.1,.3,0,0])
    expected_covariance = np.zeros((5,5))
    expected_covariance[:3,:3] = old.covariance
    expected_covariance[3,3],expected_covariance[4,4] = .36,.49
    np.testing.assert_array_equal(extended.covariance,expected_covariance)
    # Evaluate the declared image response directly in acquisition order.
    displacements = np.array([.2,-.2,.3,.2,-.4])
    weights = 1-np.exp(-displacements) if log_depth_scale else displacements
    np.testing.assert_allclose(offset,np.sum(J*weights,axis=1),atol=1e-15)
    derivative = J*np.exp(-displacements) if log_depth_scale else J
    np.testing.assert_allclose(response,derivative[:,[2,4,0,1,3]],atol=1e-15)
    np.testing.assert_array_equal(state.mean,old.mean)
    np.testing.assert_array_equal(state.covariance,old.covariance)
    np.testing.assert_array_equal(extended.jacobian[:,:3],old.jacobian)


def test_conditional_product_matches_full_joint_conditioning():
    rng = np.random.default_rng(82)
    for _ in range(40):
        def covariance(n):
            A = rng.normal(size=(n,n))
            return A@A.T+np.eye(n)*.1
        S,R,V = covariance(6),covariance(6),covariance(3)
        J,A = rng.normal(size=(6,3)),rng.normal(size=(6,3))
        mean,center,residual = rng.normal(size=3),rng.normal(size=3),rng.normal(size=6)
        state = SourceState(S,('one','two','three'),mean,V,J,np.ones(3))
        factor = SourceFactor(state.keys,A,np.ones(3),center)
        result = conditional_product(state,residual,R,factor)
        joint_mean = np.r_[np.zeros(6),mean]
        joint_covariance = np.block([[S+J@V@J.T,J@V],[V@J.T,V]])
        H,z = np.c_[np.eye(6),-A],residual-A@center
        innovation = H@joint_covariance@H.T+R
        gain = np.linalg.solve(innovation,H@joint_covariance).T
        expected_mean = joint_mean+gain@(z-H@joint_mean)
        expected_cov = joint_covariance-gain@H@joint_covariance
        posterior = result.state
        actual_mean = np.r_[result.pose_offset,posterior.mean]
        actual_cov = np.block([[posterior.marginal_covariance(),posterior.jacobian@posterior.covariance],
                              [posterior.covariance@posterior.jacobian.T,posterior.covariance]])
        np.testing.assert_allclose(actual_mean,expected_mean,atol=1e-12)
        np.testing.assert_allclose(actual_cov,expected_cov,atol=1e-12)
        r = z-H@joint_mean
        expected_log = -.5*(r@np.linalg.solve(innovation,r)+np.linalg.slogdet(innovation)[1]+6*np.log(2*np.pi))
        assert result.log_overlap == pytest.approx(expected_log,abs=1e-12)


def test_shared_bias_is_learned_once_and_retains_batch_uncertainty_floor():
    S = np.eye(6)
    S[0,0] = 4.
    state = SourceState(S)
    pose = np.zeros(6)
    J = np.zeros((6,1))
    J[0,0] = -1.
    for i in range(200):
        factor = SourceFactor(('anchor',),J,np.array([.12**2]),factor_id=f'geometry-{i}')
        observation = np.r_[2.,np.zeros(5)]
        result = conditional_product(state,observation-pose,np.eye(6)*.02**2,factor)
        pose += result.pose_offset
        state = result.state
    expected = 1/(1/4+1/(.12**2+.02**2/200))
    assert state.marginal_covariance()[0,0] == pytest.approx(expected,rel=1e-12)
    assert len(state.keys) == 1 and len(state.seen_factors) == 200
    independent = 1/(1/4+200/(.12**2+.02**2))
    assert state.marginal_covariance()[0,0] > independent*190
    # Identical delivery is not one more noisy observation.
    replay = conditional_product(state,np.ones(6)*1e6,np.eye(6)*1e-12,factor)
    assert replay.duplicate and replay.log_overlap is None
    np.testing.assert_array_equal(replay.pose_offset,np.zeros(6))
    np.testing.assert_array_equal(replay.state.covariance,state.covariance)
    np.testing.assert_array_equal(replay.state.geometry_covariance,state.geometry_covariance)


def test_equal_sensitivities_cancel_in_the_innovation_not_the_pose_marginal():
    J = np.zeros((6,1)); J[0,0] = 2.
    state = SourceState(np.eye(6)*.01,('teacher',),np.array([.2]),np.array([[.04]]),J,np.array([.04]))
    factor = SourceFactor(state.keys,J,np.array([.04]),center=state.mean)
    result = conditional_product(state,np.ones(6)*.3,np.eye(6)*.02,factor)
    np.testing.assert_allclose(result.innovation_covariance,np.eye(6)*.03)
    np.testing.assert_array_equal(result.state.mean,state.mean)
    np.testing.assert_array_equal(result.state.covariance,state.covariance)
    assert result.state.marginal_covariance()[0,0] > .16


def test_extension_preserves_old_posterior_and_does_not_reapply_its_prior():
    J = np.zeros((6,1)); J[0,0] = 1.
    state = SourceState(np.eye(6)*.1)
    factor = SourceFactor(('image-a',),J,np.array([.09]))
    posterior = conditional_product(state,np.ones(6)*.2,np.eye(6)*.01,factor).state
    extension = SourceFactor(('image-b','image-a'),np.c_[J,J],np.array([.04,.09]))
    expanded,_,_ = posterior.expand(extension)
    assert expanded.keys == ('image-a','image-b')
    assert expanded.covariance[0,0] == posterior.covariance[0,0] < .09
    assert expanded.covariance[1,1] == .04 and expanded.covariance[0,1] == 0.
    with pytest.raises(ValueError,match='different prior'):
        posterior.expand(SourceFactor(('image-a',),J,np.array([.01])))
    seeded,offset = posterior.seed(extension,np.eye(6)*.2)
    np.testing.assert_array_equal(seeded.covariance,expanded.covariance)
    np.testing.assert_array_equal(seeded.mean,expanded.mean)
    assert len(seeded.keys)==2


def test_invalid_conditional_covariance_is_not_silently_used_as_information():
    with pytest.raises(ValueError,match='positive semidefinite'):
        SourceState(-np.eye(6))
    factor = SourceFactor((),np.empty((6,0)),np.empty(0))
    with pytest.raises(ValueError,match='positive semidefinite'):
        conditional_product(SourceState(np.eye(6)),np.zeros(6),-np.eye(6)*.1,factor)
