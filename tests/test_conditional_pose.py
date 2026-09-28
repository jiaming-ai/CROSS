import json

import numpy as np
import pytest

from cross.core.conditional import SourceFactor, SourceState
from cross.core.conditional_pose import ConditionalPose, adjoint, compose, exp, inverse, log, right_jacobian


def model(keys, J, center=None):
    return ConditionalPose(np.eye(6)*.002, SourceFactor(keys,J,np.full(len(keys),.0144),center))


def test_se3_response_matches_independent_finite_differences():
    rng = np.random.default_rng(712)
    for _ in range(30):
        x = rng.normal(size=6)*.6
        numerical = np.column_stack([(log(inverse(exp(x))@exp(x+np.eye(6)[i]*1e-6))-
                                      log(inverse(exp(x))@exp(x-np.eye(6)[i]*1e-6)))/2e-6 for i in range(6)])
        np.testing.assert_allclose(log(exp(x)),x,atol=1e-12)
        np.testing.assert_allclose(right_jacobian(x),numerical,atol=3e-10)


def test_conditional_convolution_reuses_sources_and_transports_the_tangent():
    rng = np.random.default_rng(731)
    a,b = exp(rng.normal(size=6)),exp(rng.normal(size=6)*.3)
    ja,jb = rng.normal(size=(6,2)),rng.normal(size=(6,2))
    first,second = model(('shared','a'),ja),model(('b','shared'),jb)
    belief = SourceState(np.zeros((6,6)))
    pose,message = compose(a,first,b,second,belief)
    np.testing.assert_allclose(pose,a@b,atol=1e-15)
    assert message.factor.keys == ('shared','a','b')
    for i,key in enumerate(message.factor.keys):
        da = ja[:,first.factor.keys.index(key)] if key in first.factor.keys else np.zeros(6)
        db = jb[:,second.factor.keys.index(key)] if key in second.factor.keys else np.zeros(6)
        plus = a@exp(da*1e-6)@b@exp(db*1e-6)
        minus = a@exp(-da*1e-6)@b@exp(-db*1e-6)
        numerical = (log(inverse(pose)@plus)-log(inverse(pose)@minus))/2e-6
        np.testing.assert_allclose(message.factor.jacobian[:,i],numerical,atol=4e-10)
    A = adjoint(inverse(b))
    np.testing.assert_allclose(message.geometry_covariance,A@first.geometry_covariance@A.T+second.geometry_covariance)
    assert 'covariance' not in message.factor.record()


def test_source_cancellation_survives_saved_message_composition():
    J = np.zeros((6,1)); J[0,0] = -2.
    node = model(('image',),J)
    relative = model(('image',),-J)
    _,message = compose(np.eye(4),node,np.eye(4),relative,SourceState(np.zeros((6,6))))
    np.testing.assert_array_equal(message.factor.jacobian,np.zeros((6,1)))
    belief,_ = SourceState(np.zeros((6,6))).with_pose(message)
    assert belief.covariance[0,0] == .0144
    np.testing.assert_allclose(belief.marginal_covariance(),node.geometry_covariance+relative.geometry_covariance)


def test_message_at_posterior_mean_rebases_without_updating_the_bias_belief():
    rng = np.random.default_rng(818)
    node = model(('image',),rng.normal(size=(6,1)),np.array([-.05]))
    belief,_,_ = SourceState(np.eye(6)).expand(node.factor)
    belief.mean[0] = .15
    belief.covariance[0,0] = .006
    pose,message,after = node.at(exp(rng.normal(size=6)),belief)
    assert message.factor.center[0] == .15
    np.testing.assert_array_equal(after.mean,belief.mean)
    np.testing.assert_array_equal(after.covariance,belief.covariance)
    # A node's model can be serialized without serializing the bias posterior.
    restored = ConditionalPose.from_record(json.loads(json.dumps(message.record())))
    np.testing.assert_array_equal(restored.factor.jacobian,message.factor.jacobian)
    np.testing.assert_array_equal(restored.geometry_covariance,message.geometry_covariance)
    bias = SourceState.from_record(json.loads(json.dumps(after.record())))
    np.testing.assert_array_equal(bias.covariance,after.covariance)
    empty = SourceState.from_record(json.loads(json.dumps(SourceState(np.eye(6)).record())))
    assert empty.jacobian.shape == (6,0)


def test_invalid_saved_source_covariance_is_rejected():
    node = model(('a','b'),np.zeros((6,2)))
    belief,_,_ = SourceState(np.eye(6)).expand(node.factor)
    record = belief.record(); record['covariance'] = [[1.,2.],[2.,1.]]
    with pytest.raises(ValueError,match='positive semidefinite'):
        SourceState.from_record(record)
