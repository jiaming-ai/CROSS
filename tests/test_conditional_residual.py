"""Residual-coordinate conditioning, including an unobservable shared frame."""
import numpy as np
import pytest

from cross.core.conditional import SourceFactor,SourceState,conditional_product
from cross.core.conditional_pose import adjoint,exp,inverse,log,residual_product,right_jacobian


def test_residual_jacobians_match_independent_group_perturbations():
    rng=np.random.default_rng(106)
    for _ in range(12):
        P,O=exp(rng.normal(size=6)*.4),exp(rng.normal(size=6)*.4)
        r=log(inverse(P)@O)
        F=np.linalg.inv(right_jacobian(-r));G=np.linalg.inv(right_jacobian(r))
        for i in range(6):
            d=np.eye(6)[i]*1e-6
            prior=(log(inverse(P@exp(d))@O)-log(inverse(P@exp(-d))@O))/2e-6
            observation=(log(inverse(P)@O@exp(d))-log(inverse(P)@O@exp(-d)))/2e-6
            np.testing.assert_allclose(prior,-F[:,i],atol=1e-8)
            np.testing.assert_allclose(observation,G[:,i],atol=1e-8)


def test_shared_rigid_frame_cannot_be_observed_by_a_relative_pose_residual():
    P=exp(np.array([.3,-.2,.4,.1,.2,-.1]))
    O=P@exp(np.array([.7,.3,-.2,.2,-.1,.3]))
    r=log(inverse(P)@O)
    state=SourceState(np.eye(6)*.01,tuple('abcdef'),np.zeros(6),np.eye(6)*.25,
                      adjoint(inverse(P)),np.ones(6)*.25)
    factor=SourceFactor(state.keys,adjoint(inverse(O)),state.prior_variances)
    result=residual_product(state,r,np.eye(6)*.01,factor)
    np.testing.assert_allclose(result.state.mean,state.mean,atol=1e-12)
    np.testing.assert_allclose(result.state.covariance,state.covariance,atol=1e-12)
    # Old one-sided transport gives a false >1-unit update to this gauge.
    G=np.linalg.inv(right_jacobian(r))
    old=conditional_product(state,r,G@np.eye(6)*.01@G.T,
        SourceFactor(state.keys,G@factor.jacobian,state.prior_variances))
    assert np.linalg.norm(old.state.mean)>1.


def test_residual_product_matches_joint_batch_gaussian_with_numeric_jacobians():
    rng=np.random.default_rng(923)
    for _ in range(12):
        def covariance(n):
            A=rng.normal(size=(n,n))*.1
            return A@A.T+np.eye(n)*.05
        P,O=exp(rng.normal(size=6)*.3),exp(rng.normal(size=6)*.3)
        r=log(inverse(P)@O)
        J,A=rng.normal(size=(6,3))*.1,rng.normal(size=(6,3))*.1
        S,R,V,Q=covariance(6),covariance(6),covariance(3),covariance(6)*.1
        state=SourceState(S,('a','b','c'),np.zeros(3),V,J,np.ones(3))
        result=residual_product(state,r,R,SourceFactor(state.keys,A,np.ones(3)),Q)
        F,G=np.zeros((6,6)),np.zeros((6,6))
        for i in range(6):
            d=np.eye(6)[i]*1e-6
            F[:,i]=-(log(inverse(P@exp(d))@O)-log(inverse(P@exp(-d))@O))/2e-6
            G[:,i]=(log(inverse(P)@O@exp(d))-log(inverse(P)@O@exp(-d)))/2e-6
        joint=np.block([[S+Q+J@V@J.T,J@V],[V@J.T,V]])
        H=np.c_[F,-G@A]
        innovation=H@joint@H.T+G@R@G.T
        gain=np.linalg.solve(innovation,H@joint).T
        expected_mean=gain@r
        expected_cov=joint-gain@H@joint
        posterior=result.state
        actual_cov=np.block([[posterior.marginal_covariance(),posterior.jacobian@posterior.covariance],
                             [posterior.covariance@posterior.jacobian.T,posterior.covariance]])
        np.testing.assert_allclose(np.r_[result.pose_offset,posterior.mean],expected_mean,atol=1e-8)
        np.testing.assert_allclose(actual_cov,expected_cov,atol=1e-8)
        np.testing.assert_allclose(result.innovation_covariance,innovation,atol=1e-8)


def test_zero_residual_retains_the_euclidean_conditional_product():
    state=SourceState(np.eye(6),('a',),np.zeros(1),np.eye(1),np.ones((6,1)),np.ones(1))
    factor=SourceFactor(state.keys,np.arange(6).reshape(6,1),np.ones(1))
    old=conditional_product(state,np.zeros(6),np.eye(6)*.1,factor,np.eye(6)*.02)
    new=residual_product(state,np.zeros(6),np.eye(6)*.1,factor,np.eye(6)*.02)
    np.testing.assert_array_equal(new.pose_offset,old.pose_offset)
    np.testing.assert_array_equal(new.state.covariance,old.state.covariance)
    np.testing.assert_array_equal(new.state.geometry_covariance,old.state.geometry_covariance)
    assert new.log_overlap==old.log_overlap


def test_residual_product_rejects_an_unevaluated_source_center():
    state=SourceState(np.eye(6),('a',),np.ones(1),np.eye(1),np.ones((6,1)),np.ones(1))
    factor=SourceFactor(state.keys,np.zeros((6,1)),np.ones(1))
    with pytest.raises(ValueError,match='Evaluate and align'):
        residual_product(state,np.zeros(6),np.eye(6),factor)
