import numpy as np
import torch

from cross.core.types import Camera
from cross.cv.pose_est_ff import FFPrediction, covisibility_scores
from cross.utils.camera import get_transforms_ff


def test_transform_ff_kitti_like():
    K = np.array([[700.0, 0, 620.0], [0, 700.0, 190.0], [0, 0, 1.0]])
    cam = Camera(K=K.copy(), frame_width=1242, frame_height=375)
    rgb_tf, depth_tf = get_transforms_ff(cam, 512)
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    out = rgb_tf(img)
    assert out.shape == (3, cam.frame_height, cam.frame_width)
    assert cam.frame_width == 512 and cam.frame_height % 16 == 0
    # aspect crop to 0.5 -> 750 px wide crop -> scale 512/750
    s = 512 / 750
    assert abs(cam.fx - 700 * s) < 1e-6
    assert abs(cam.px - (620 - (1242 - 750) / 2) * s) < 1e-6
    d = depth_tf(torch.zeros(1, 375, 1242))
    assert d.shape == (1, cam.frame_height, cam.frame_width)


def test_transform_ff_square():
    cam = Camera(K=np.array([[320.0, 0, 319.5], [0, 320.0, 319.5], [0, 0, 1.0]]), frame_width=640, frame_height=640)
    rgb_tf, _ = get_transforms_ff(cam, 512)
    assert rgb_tf(np.zeros((640, 640, 3), dtype=np.uint8)).shape == (3, 512, 512)
    assert abs(cam.fx - 256.0) < 1e-6


def _synthetic_pred(device="cpu"):
    H, W = 64, 96
    K = np.array([[80.0, 0, W / 2], [0, 80.0, H / 2], [0, 0, 1]])
    # a fronto-parallel plane at depth 5 seen from two cameras displaced by 0.5 m in x
    depth0 = torch.full((H, W), 5.0)
    depth1 = torch.full((H, W), 5.0)
    c2w = np.repeat(np.eye(4)[None], 3, 0)
    c2w[1, 0, 3] = 0.5
    c2w[2, 2, 3] = 50.0            # camera far beyond the plane: no overlap
    depth2 = torch.full((H, W), 5.0)
    return FFPrediction(c2w=c2w, K=np.stack([K, K, K]), depth=torch.stack([depth0, depth1, depth2]),
                        depth_conf=None, hw=(H, W))


def test_covisibility_overlap_vs_none():
    pred = _synthetic_pred()
    s = covisibility_scores(pred, [1, 2], 0, grid=32)
    assert s[0] > 0.8          # small lateral shift of a plane: nearly all points consistent
    assert s[1] < 0.05         # camera behind the plane: nothing projects consistently


def test_principal_point_rotation():
    from cross.cv.pose_est_ff import principal_point_rotation
    K_centred = np.array([[320.0, 0, 319.5], [0, 320.0, 239.5], [0, 0, 1]])
    assert np.allclose(principal_point_rotation(K_centred, 640, 480), np.eye(3))
    K = np.array([[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1]])        # KITTI 00-02
    R = principal_point_rotation(K, 1241, 376)
    ray = np.array([(620.0 - 607.1928) / 718.856, (187.5 - 185.2157) / 718.856, 1.0])
    assert np.allclose(R @ [0, 0, 1.0], ray / np.linalg.norm(ray))           # model optical axis -> image-centre ray
    assert abs(np.degrees(np.arccos((np.trace(R) - 1) / 2)) - 1.04) < 0.01
    assert np.allclose(R @ R.T, np.eye(3))


def test_map_anchor_pair_covisibility_filter():
    from cross.cv.pose_est_ff import filter_map_anchors
    from cross.cv.stereo_scale import ScaleAnchor
    pred = _synthetic_pred()           # views 0 and 1 overlap, view 2 overlaps neither
    T = np.eye(4)
    anchors = [ScaleAnchor(1, 2, T, kind="map"), ScaleAnchor(0, 1, T, kind="map"), ScaleAnchor(0, 2, T, kind="map"),
               ScaleAnchor(0, 1, T, kind="odom")]
    kept = filter_map_anchors(pred, anchors, 0.15, max_pairs=8, grid=32)
    assert [(a.kind, a.idx_a, a.idx_b) for a in kept] == [("odom", 0, 1), ("map", 0, 1)]
    assert len(filter_map_anchors(pred, anchors[:2], 0.15, max_pairs=0, grid=32)) == 0
    # the covisibility head's matrix when there is no depth for the references (covis_source=head)
    pred.depth = pred.depth[:0]
    pred.covis = np.array([[1, .6, .0], [.5, 1, .1], [.0, .2, 1]])
    kept = filter_map_anchors(pred, anchors, 0.15, max_pairs=8, source="head")
    assert [(a.kind, a.idx_a, a.idx_b) for a in kept] == [("odom", 0, 1), ("map", 0, 1)]


def test_pairwise_covisibility_matches_per_destination_scores():
    from cross.cv.pose_est_ff import pairwise_covisibility
    pred = _synthetic_pred()
    rng = np.random.default_rng(0)
    pred.depth_conf = torch.from_numpy(rng.uniform(0.5, 2.0, pred.depth.shape).astype(np.float32))
    views = [0, 1, 2]
    C = pairwise_covisibility(pred, views, grid=32)
    for b in views:
        src = [v for v in views if v != b]
        assert np.allclose(C[src, b], covisibility_scores(pred, src, b, grid=32), atol=1e-6)
    assert np.allclose(np.diag(C), 1.0)


def test_set_camera_registers_calibrated_focal_and_principal_point():
    """set_camera gives the backend the calibrated focal (canonical-camera scale heads) and sets the principal-point
    correction (one definition: a second `def set_camera` once silently dropped the focal part)."""
    from types import SimpleNamespace
    from cross.core.config import FeedForwardConfig
    from cross.cv.pose_est_ff import PoseEstFeedForward, principal_point_rotation
    K = np.array([[300.0, 0, 270.0], [0, 300.0, 180.0], [0, 0, 1]])
    for scale_focal, ppc in (("calibrated", True), ("predicted", True), ("calibrated", False)):
        est = PoseEstFeedForward.__new__(PoseEstFeedForward)        # no model: only the camera bookkeeping
        est.config = FeedForwardConfig(scale_focal=scale_focal, principal_point_correction=ppc)
        est.backend = SimpleNamespace()
        est._R_model_to_cam = None
        est.set_camera(K, 518, 350)
        assert est._focal_norm == 300.0 / 518
        assert getattr(est.backend, "calib_focal", None) == (300.0 / 518 if scale_focal == "calibrated" else None)
        if ppc:
            assert np.allclose(est._R_model_to_cam[:3, :3], principal_point_rotation(K, 518, 350).T)
        else:
            assert est._R_model_to_cam is None


def test_no_duplicate_method_definitions():
    """A second `def` of a name in one class / module body silently replaces the first."""
    import ast
    import pathlib
    import warnings
    root = pathlib.Path(__file__).resolve().parents[1]
    dups = []
    for path in [*(root / "cross").rglob("*.py"), *(root / "vggt_ft").rglob("*.py")]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef)):
                continue
            seen = set()
            for b in node.body:
                if not isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if any(k in ast.unparse(d) for d in b.decorator_list for k in ("setter", "deleter", "overload", "register")):
                    continue
                if b.name in seen:
                    dups.append(f"{path.relative_to(root)}:{b.lineno} {getattr(node, 'name', '<module>')}.{b.name}")
                seen.add(b.name)
    assert not dups, dups
