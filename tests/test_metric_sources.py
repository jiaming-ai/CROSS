import pickle

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cross.mono.metric_sources import (
    TranslationResponse, identify_prediction, pnp_scale_response, relative_scale_response,
)


def pose(xyz, rotvec=(0., 0., 0.)):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    out[:3, 3] = xyz
    return out


def test_source_identity_is_content_not_frame_session_or_object_identity():
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    calibration = np.array([[20.,0.,8.],[0.,20.,6.],[0.,0.,1.]])
    options = dict(model_id="teacher", revision="weights-sha", resolution=504, session_id="first")
    first = identify_prediction(image, calibration, **options)
    again = identify_prediction(image.copy(), calibration.astype(np.float32), **dict(options, session_id="second"))
    assert first.source_id == again.source_id
    assert first.session_id != again.session_id
    assert first == pickle.loads(pickle.dumps(first))
    for replacement in [dict(revision="different"), dict(resolution=336), dict(model_id="other")]:
        assert identify_prediction(image, calibration, **dict(options, **replacement)).source_id != first.source_id
    assert identify_prediction(image+1, calibration, **options).source_id != first.source_id
    new_k = calibration.copy()
    new_k[0,0] += 1
    assert identify_prediction(image, new_k, **options).source_id != first.source_id
    with pytest.raises(ValueError, match="pinned"):
        identify_prediction(image, calibration, **dict(options, revision=None))


def test_signed_reuse_and_return_motion_cancel_before_marginalization():
    first = TranslationResponse().with_displacement("anchor", [1., 0., 0.])
    second = TranslationResponse().with_displacement("anchor", [3., 0., 0.])
    increment = relative_scale_response(pose([1,0,0]), first, pose([3,0,0]), second)
    assert increment == {"anchor": [-2.,0.,0.,0.,0.,0.]}
    # A shared 10% bias contributes (2*.1)^2, not the sum of the absolute
    # endpoint variances (1*.1)^2+(3*.1)^2 and not an unsigned envelope.
    assert (increment["anchor"][0]*.1)**2 == pytest.approx(.04)
    assert relative_scale_response(pose([1,0,0]), first, pose([1,0,0]), first) == {}
    returned = first.with_displacement("anchor", [-1.,0.,0.])
    assert not returned.terms
    assert first.record() == {"anchor": [-1.,0.,0.]}


def test_multisource_right_tangent_matches_finite_differences_and_queue_composition():
    a = pose([.2,-.1,.3], [.15,.4,-.2])
    b = pose([1.2,.3,1.7], [-.2,.5,.3])
    c = pose([.4,.8,2.1], [.3,-.4,.2])
    ra = TranslationResponse()
    rb = ra.with_displacement("one", b[:3,3]-a[:3,3])
    rc = rb.with_displacement("two", c[:3,3]-b[:3,3])
    direct = relative_scale_response(a, ra, c, rc)
    first = relative_scale_response(a, ra, b, rb)
    second = relative_scale_response(b, rb, c, rc)
    for source in direct:
        # Chain right tangent Jacobians: Ad(T_bc^-1) J_ab + J_bc.
        expected = c[:3,:3].T @ b[:3,:3] @ np.asarray(first.get(source, [0.]*6))[:3]
        expected += np.asarray(second.get(source, [0.]*6))[:3]
        np.testing.assert_allclose(direct[source][:3], expected, atol=1e-14)
        epsilon = 1e-6
        plus = rc.translation_at(c[:3,3], {source: epsilon})
        minus = rc.translation_at(c[:3,3], {source: -epsilon})
        finite_difference = c[:3,:3].T @ ((plus-minus)/(2*epsilon))
        np.testing.assert_allclose(direct[source][:3], finite_difference, rtol=1e-8, atol=1e-9)
    assert rc == pickle.loads(pickle.dumps(rc))
    np.testing.assert_allclose(pnp_scale_response(np.linalg.inv(a)@b)[:3], first["one"][:3], atol=1e-14)
    # Independent sources stay distinct even if their displacements cancel.
    assert len(ra.with_displacement("one", [1,0,0]).with_displacement("two", [-1,0,0]).terms) == 2
