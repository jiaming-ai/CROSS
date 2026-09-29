"""Finite source actions must survive mean correction and later evaluation."""
import numpy as np
import pytest

from cross.core.conditional import SourceState,SourceFactor
from cross.core.conditional_pose import (ConditionalPose,adjoint,exp,inverse,log,
    residual_product,right_jacobian,retract_frozen_response)


def test_frozen_response_is_the_derivative_of_a_composed_pose_family():
    rng=np.random.default_rng(1742)
    keys=tuple('geometry:g:'+str(i) for i in range(6))+('image:a','image:b')
    for _ in range(16):
        J0=rng.normal(size=(6,8));J1=rng.normal(size=(6,8));d=rng.normal(size=6)*.15
        expected=retract_frozen_response(keys,J1,d,J0)
        P=exp(rng.normal(size=6));nominal=P@exp(d)
        def family(b):
            correction=(J1[:,:6]-J0[:,:6])@b[:6]+J1[:,6:]@b[6:]
            return P@exp(J0[:,:6]@b[:6])@exp(d+correction)
        numeric=np.empty_like(J1)
        for i in range(8):
            h=np.eye(8)[i]*1e-6
            numeric[:,i]=(log(inverse(nominal)@family(h))-log(inverse(nominal)@family(-h)))/2e-6
        np.testing.assert_allclose(expected,numeric,atol=2e-8)


def test_evaluating_metric_bias_retains_the_shared_frozen_rigid_action():
    P=exp(np.array([2.,-.5,1.,.1,-.2,.3]))
    keys=tuple('geometry:g:'+str(i) for i in range(6))+('image:a','image:b')
    metric=np.array([[1.,0.],[.2,.1],[0.,-.5],[0.,0.],[0.,0.],[0.,0.]])
    J=np.c_[adjoint(inverse(P)),metric]
    mean=np.r_[np.zeros(6),.3,-.2]
    variance=np.r_[np.ones(6)*10000.,.01,.01]
    belief=SourceState(np.eye(6)*.01,keys,mean,np.diag(variance),np.zeros((6,8)),variance)
    model=ConditionalPose(np.eye(6)*.02,SourceFactor(keys,J,variance))
    pose,moved,_=model.at(P,belief)
    same,known=model.at_known(P,belief)
    np.testing.assert_allclose(pose,same,atol=1e-12)
    np.testing.assert_allclose(moved.factor.jacobian,known.factor.jacobian,atol=1e-12)
    np.testing.assert_allclose(moved.factor.jacobian[:,:6],adjoint(inverse(pose)),atol=1e-12)
    np.testing.assert_allclose(moved.factor.jacobian[:,6:],right_jacobian(J@mean)@metric,atol=1e-12)


@pytest.mark.parametrize('strength',[1.,100.])
def test_repeated_relative_updates_cannot_observe_a_common_rigid_action(strength):
    rng=np.random.default_rng(728)
    keys=tuple('geometry:g:'+str(i) for i in range(6))
    P=exp(np.array([1.,-2.,.4,.1,.4,-.2]));pose=P.copy()
    V=np.eye(6)*strength**2
    state=SourceState(np.eye(6)*.02,keys,np.zeros(6),V,adjoint(inverse(P)),np.full(6,strength**2))
    for i in range(12):
        observed=P@exp(rng.normal(size=6)*.12)
        f=SourceFactor(keys,adjoint(inverse(observed)),state.prior_variances,np.zeros(6),str(i))
        old=state.jacobian.copy()
        value=residual_product(state,log(inverse(pose)@observed),np.eye(6)*.02,f,np.eye(6)*.001,keys)
        d=value.pose_offset;state=value.state;T=right_jacobian(d)
        state.jacobian=retract_frozen_response(keys,state.jacobian,d,old)
        state.geometry_covariance=T@state.geometry_covariance@T.T
        state.geometry_covariance=(state.geometry_covariance+state.geometry_covariance.T)/2
        pose=pose@exp(d)
        np.testing.assert_allclose(state.jacobian,adjoint(inverse(pose)),atol=1e-8)
        np.testing.assert_array_equal(state.covariance,V)
        np.testing.assert_array_equal(state.mean,np.zeros(6))


def test_active_only_sources_retain_the_declared_additive_response():
    rng=np.random.default_rng(402)
    J=rng.normal(size=(6,3));before=rng.normal(size=(6,3));d=rng.normal(size=6)*.2
    expected=right_jacobian(d)@J
    np.testing.assert_array_equal(retract_frozen_response(('a','b','c'),J,d,before),expected)
    np.testing.assert_array_equal(retract_frozen_response(('geometry:a','b','c'),J,np.zeros(6),before),J)
