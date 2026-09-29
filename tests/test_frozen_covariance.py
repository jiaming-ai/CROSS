"""Structured Schmidt results must equal an independent augmented Joseph filter."""
import json

import numpy as np
import pytest

from cross.core.conditional import SourceFactor, SourceState
from cross.core.frozen_covariance import FrozenDiagonalCovariance
from cross.core.schmidt import schmidt_product, _response


def distribution(rng,a,g,deterministic=False):
    B=rng.normal(size=(a,g))*.015
    L=rng.normal(size=(a,a))*.04
    Q=L@L.T+np.eye(a)*.02
    if deterministic and a:
        B[-1]=0;Q[-1]=0;Q[:,-1]=0
    diagonal=rng.uniform(.5,1.5,size=g)
    A=Q+(B/diagonal)@B.T
    A=(A+A.T)/2
    V=np.block([[A,B],[B.T,np.diag(diagonal)]])
    permutation=rng.permutation(a+g)
    V=V[np.ix_(permutation,permutation)]
    frozen=np.flatnonzero(permutation>=a)
    keys=tuple(('geometry:factor:' if index>=a else 'image:')+str(index) for index in permutation)
    return V,frozen,keys


@pytest.mark.parametrize('a,g,deterministic',[(0,18,False),(4,24,False),(4,24,True),(12,48,False)])
def test_repeated_schmidt_matches_joint_joseph(a,g,deterministic):
    rng=np.random.default_rng(8717)
    V,frozen,keys=distribution(rng,a,g,deterministic)
    S=np.eye(6)*.03;J=rng.normal(size=(6,a+g))*.2
    mean=rng.normal(size=a+g)*.1
    prior_variances=V.diagonal().copy()
    state=SourceState(S,keys,mean,FrozenDiagonalCovariance.from_dense(V,frozen),J,prior_variances)
    joint=np.block([[S+J@V@J.T,J@V],[V@J.T,V]])
    frozen_keys=tuple(keys[i] for i in frozen)
    for step in range(20):
        A=rng.normal(size=(6,a+g))*.2;center=rng.normal(size=a+g)*.03
        residual=rng.normal(size=6)*.1;R=np.eye(6)*.02;Q=np.eye(6)*.001
        factor=SourceFactor(keys,A,prior_variances,center,'pair:'+str(step))
        result=schmidt_product(state,residual,R,factor,Q,frozen_keys)
        prior_joint=joint.copy();prior_joint[:6,:6]+=Q
        H=np.c_[np.eye(6),-A]
        innovation=H@prior_joint@H.T+R
        gain=np.linalg.solve(innovation,H@prior_joint).T
        gain[6+frozen]=0
        update=residual+A@(mean-center)
        expected_mean=np.r_[np.zeros(6),mean]+gain@update
        remainder=np.eye(len(joint))-gain@H
        joint=remainder@prior_joint@remainder.T+gain@R@gain.T
        mean=expected_mean[6:]
        post=result.state;dense=post.covariance.to_dense()
        actual=np.block([[post.marginal_covariance(),post.jacobian@post.covariance],
            [post.covariance@post.jacobian.T,dense]])
        np.testing.assert_allclose(actual,joint,rtol=2e-9,atol=1e-10)
        np.testing.assert_allclose(np.r_[result.pose_offset,post.mean],expected_mean,rtol=1e-9,atol=1e-10)
        expected_log=-.5*(update@np.linalg.solve(innovation,update)+np.linalg.slogdet(innovation)[1]+6*np.log(2*np.pi))
        np.testing.assert_allclose(result.log_overlap,expected_log,rtol=1e-9,atol=1e-10)
        np.testing.assert_array_equal(dense[np.ix_(frozen,frozen)],V[np.ix_(frozen,frozen)])
        np.testing.assert_array_equal(post.mean[frozen],state.mean[frozen])
        duplicate=schmidt_product(post,residual,R,factor,Q,frozen_keys)
        assert duplicate.duplicate and duplicate.log_overlap is None
        assert duplicate.state.record()==post.record()
        state=post


def test_interleaved_extension_preserves_all_source_correlations_and_persistence():
    rng=np.random.default_rng(81)
    V,frozen,keys=distribution(rng,3,7)
    matrix=FrozenDiagonalCovariance.from_dense(V,frozen)
    state=SourceState(np.eye(6),keys,np.zeros(10),matrix,np.ones((6,10)),V.diagonal())
    extra=SourceFactor(('image:new','geometry:factor:new:0'),np.ones((6,2)),np.array([.02,1.]))
    extended,_,_=state.expand(extra)
    expected=np.zeros((12,12));expected[:10,:10]=V;expected[10,10]=.02;expected[11,11]=1
    np.testing.assert_array_equal(extended.covariance.to_dense(),expected)
    payload=json.loads(json.dumps(extended.record()))
    assert payload['version']==2
    restored=SourceState.from_record(payload)
    assert restored.record()==extended.record()
    assert restored.covariance.nbytes < expected.nbytes
    copied=restored.copy();copied.covariance.cross[:]=0
    np.testing.assert_array_equal(restored.covariance.to_dense(),expected)
    # No caller can accidentally materialize a nuisance-by-nuisance matrix.
    with pytest.raises(TypeError,match='Implicit dense'):
        np.asarray(restored.covariance)


def test_singular_schur_response_preserves_joint_distribution():
    rng=np.random.default_rng(99)
    V,frozen,_=distribution(rng,5,17,True)
    cross=rng.normal(size=(6,22))@V
    structured=FrozenDiagonalCovariance.from_dense(V,frozen)
    J=_response(structured,cross);reference=_response(V,cross)
    np.testing.assert_allclose(J@structured,cross,atol=1e-10)
    np.testing.assert_allclose(J@cross.T,reference@cross.T,atol=1e-9)
    np.testing.assert_allclose(structured@cross.T,V@cross.T,atol=1e-10)


def test_no_approximation_of_correlated_nuisance_or_invalid_persisted_covariance():
    V=np.eye(5);V[2,3]=V[3,2]=.1
    with pytest.raises(ValueError,match='correlated frozen'):
        FrozenDiagonalCovariance.from_dense(V,[2,3,4])
    matrix=FrozenDiagonalCovariance.from_dense(np.eye(5),[2,3,4])
    record=matrix.record();record['cross'][0][0]=2
    with pytest.raises(ValueError,match='positive semidefinite'):
        FrozenDiagonalCovariance.from_record(record)


def test_graph_noise_partition_cannot_change_on_replay():
    keys=('image:a','geometry:factor:edge:0')
    matrix=FrozenDiagonalCovariance.from_dense(np.eye(2),[1])
    state=SourceState(np.eye(6),keys,np.zeros(2),matrix,np.zeros((6,2)),np.ones(2))
    with pytest.raises(ValueError,match='exactly its diagonal'):
        schmidt_product(state,np.zeros(6),np.eye(6),
            SourceFactor(keys,np.zeros((6,2)),np.ones(2)),frozen_keys=keys)
    with pytest.raises(ValueError,match='match persistent'):
        SourceState(np.eye(6),('image:a','image:b'),np.zeros(2),matrix,np.zeros((6,2)),np.ones(2))
