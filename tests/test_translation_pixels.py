"""A metric algebraic fit must not destroy feasible pixel consensus."""
import numpy as np

from cross.mono.translation import translation_given_rotation


def two_depth_fixture():
    K = np.array([[520., 0., 320.], [0., 520., 240.], [0., 0., 1.]])
    xy = np.array([(x, y) for y in range(70, 421, 25) for x in range(70, 571, 25)], float)
    far = np.random.default_rng(7).permutation(len(xy))[:len(xy)//5]
    depth = np.full(len(xy), 2.)
    depth[far] = 30.
    points = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(K).T * depth[:, None]
    pixels = xy.copy()
    pixels[far, 0] += 2.5
    return K, points, pixels, xy


def test_depth_range_does_not_destroy_an_all_inlier_feasible_pose():
    K, points, pixels, truth = two_depth_fixture()
    # Exact depth/rotation; zero translation already meets the declared3px gate.
    # The old unweighted metric refinement rejects this315-point consensus.
    assert np.max(np.linalg.norm(pixels-truth, axis=1)) == 2.5
    result = translation_given_rotation(points, pixels, K, np.eye(3))
    assert result is not None
    translation, inliers, error = result
    assert len(inliers) == len(points) and error < .1
    assert np.linalg.norm(translation) < .001


def test_noisy_pixel_refinement_preserves_common_metric_scale_response():
    K, points, pixels, _ = two_depth_fixture()
    translation, inliers, error = translation_given_rotation(points, pixels, K, np.eye(3))
    for scale in (.5, 2., 5.):
        result = translation_given_rotation(points*scale, pixels, K, np.eye(3))
        assert result is not None
        np.testing.assert_allclose(result[0], translation*scale, rtol=1e-7, atol=1e-10)
        np.testing.assert_array_equal(result[1], inliers)
        np.testing.assert_allclose(result[2], error, rtol=1e-7, atol=1e-10)
