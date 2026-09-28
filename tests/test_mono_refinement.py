"""Geometry contracts used by the fast and retrieved-pair matchers."""
from types import MethodType

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cross.mono.refinement import XFeatRefiner


def geometry_fixture():
    rng = np.random.default_rng(42)
    K = np.array([[520., 0., 320.], [0., 520., 240.], [0., 0., 1.]])
    xy = np.array([(x, y) for y in range(60, 421, 40) for x in range(60, 581, 40)], dtype=float)
    z = rng.uniform(1.5, 5., len(xy))
    depth = np.full((480, 640), np.nan)
    depth[xy[:, 1].astype(int), xy[:, 0].astype(int)] = z
    xyz = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(K).T * z[:, None]
    rvec, tvec = np.array([.02, -.04, .01]), np.array([.12, .02, .04])
    pixels = cv2.projectPoints(xyz, rvec, tvec, K, None)[0].reshape(-1, 2)
    descriptors = torch.as_tensor(rng.normal(size=(len(xy), 64)), dtype=torch.float32)
    descriptors /= descriptors.norm(dim=1, keepdim=True)
    permutation = rng.permutation(len(xy))
    ref = dict(keypoints=torch.tensor(xy), descriptors=descriptors, shape=(480, 640))
    cur = dict(keypoints=torch.tensor(pixels[permutation]), descriptors=descriptors[permutation], shape=(480, 640))
    refiner = XFeatRefiner.__new__(XFeatRefiner)
    refiner.K, refiner.subpixel, refiner.rotation_selection, refiner.matcher = K, False, False, "mnn"
    expected = np.eye(4)
    expected[:3, :3], expected[:3, 3] = cv2.Rodrigues(rvec)[0], tvec
    return refiner, ref, cur, depth, np.linalg.inv(expected), pixels


def test_descriptor_permutation_keeps_pixel_pairing_and_pose_direction():
    refiner, ref, current, depth, expected, _ = geometry_fixture()
    cv2.setRNGSeed(0)
    pose, count, error = refiner.estimate(ref, current, depth)
    np.testing.assert_allclose(pose, expected, atol=1e-6)
    assert count == len(ref["keypoints"]) and error < 1e-5


def test_replacement_matcher_still_passes_spatial_and_consensus_checks():
    refiner, ref, current, depth, expected, pixels = geometry_fixture()
    xy = ref["keypoints"].numpy()
    refiner.match = MethodType(lambda self, a, b: (xy, pixels), refiner)
    pose, _, _ = refiner.estimate(ref, current, depth)
    np.testing.assert_allclose(pose, expected, atol=1e-6)
    # A different matcher is not an exemption from the geometric spatial gate.
    concentrated = pixels.copy()
    concentrated[:] = [320., 240.]
    refiner.match = MethodType(lambda self, a, b: (xy, concentrated), refiner)
    assert refiner.estimate(ref, current, depth) is None
    assert refiner.last_match_audit["reason"] in {"pnp_consensus", "poor_spatial_coverage"}
