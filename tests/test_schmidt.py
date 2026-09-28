"""Schmidt source conditioning checked against an augmented Joseph update."""
import numpy as np
import pytest

from cross.core.conditional import SourceFactor,SourceState,conditional_product
from cross.core.schmidt import schmidt_product


@pytest.mark.parametrize('deterministic_source',[False,True])
def test_schmidt_matches_joint_joseph_and_preserves_nuisance_block(deterministic_source):
    rng=np.random.default_rng(271)
    for trial in range(30):
        def covariance(n):
            A=rng.normal(size=(n,n))*.3
            return A@A.T+np.eye(n)*.1
        S,V,R,Q=covariance(6),covariance(7),covariance(6),covariance(6)*.02
        if deterministic_source:
            V[-1]=0;V[:,-1]=0
        J,A=rng.normal(size=(6,7)),rng.normal(size=(6,7))
        keys=tuple(f'source-{i}' for i in range(7));frozen=(keys[1],keys[3],keys[4])
        mean,center,residual=rng.normal(size=7),rng.normal(size=7),rng.normal(size=6)
        state=SourceState(S,keys,mean,V,J,np.ones(7))
        factor=SourceFactor(keys,A,np.ones(7),center,f'factor:{trial}')
        result=schmidt_product(state,residual,R,factor,Q,frozen)
        joint=np.block([[S+Q+J@V@J.T,J@V],[V@J.T,V]])
        H=np.c_[np.eye(6),-A]
        innovation=H@joint@H.T+R
        K=np.linalg.solve(innovation,H@joint).T
        K[[7,9,10]]=0
        B=np.eye(13)-K@H
        expected_cov=B@joint@B.T+K@R@K.T
        expected_mean=np.r_[np.zeros(6),mean]+K@(residual+A@(mean-center))
        post=result.state
        actual_cov=np.block([[post.marginal_covariance(),post.jacobian@post.covariance],
                             [post.covariance@post.jacobian.T,post.covariance]])
        np.testing.assert_allclose(actual_cov,expected_cov,atol=1e-11)
        np.testing.assert_allclose(np.r_[result.pose_offset,post.mean],expected_mean,atol=1e-11)
        np.testing.assert_array_equal(post.mean[[1,3,4]],mean[[1,3,4]])
        np.testing.assert_array_equal(post.covariance[np.ix_([1,3,4],[1,3,4])],V[np.ix_([1,3,4],[1,3,4])])
        replay=schmidt_product(post,residual,R,factor,Q,frozen)
        assert replay.duplicate
        np.testing.assert_array_equal(replay.state.covariance,post.covariance)


def test_empty_schmidt_set_keeps_full_conditioning():
    state=SourceState(np.eye(6),('a',),np.zeros(1),np.eye(1),np.ones((6,1)),np.ones(1))
    factor=SourceFactor(state.keys,np.zeros((6,1)),np.ones(1))
    old=conditional_product(state,np.ones(6),np.eye(6),factor)
    new=schmidt_product(state,np.ones(6),np.eye(6),factor)
    np.testing.assert_array_equal(old.pose_offset,new.pose_offset)
    np.testing.assert_array_equal(old.state.covariance,new.state.covariance)


def test_unknown_schmidt_source_is_rejected_before_update():
    state=SourceState(np.eye(6))
    factor=SourceFactor((),np.empty((6,0)),np.empty(0))
    with pytest.raises(ValueError,match='Every frozen source'):
        schmidt_product(state,np.zeros(6),np.eye(6),factor,frozen_keys=['absent'])
