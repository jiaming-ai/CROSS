"""Identity and local scale response of reused learned-depth predictions.

This is provenance, not a scale estimator. No covariance or pose is changed.
For a source a with log nominal/true depth bias b_a, fixed-correspondence PnP
has t_a(b_a) = exp(-b_a) t_a(0). Rotations do not depend on this scalar.
Tracking retains the signed world-translation derivative for every reused
source. Differences of these derivatives give the response of a delivered
mapping increment, even when intermediate images were dropped from its queue.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np


# Change when preprocessing, focal conversion or output conventions change.
DEPTH_POLICY = "da3-cubic14-imagenet-focal300-sky03-linear-v1"


@dataclass(frozen=True)
class MetricSource:
    source_id: str
    image_sha256: str
    model_id: str
    revision: str
    resolution: int
    session_id: str
    policy: str = DEPTH_POLICY

    def record(self):
        return asdict(self)


def identify_prediction(rgb, K, *, model_id, revision, resolution, session_id):
    """Content identity before resizing; frame/queue/session IDs are not evidence.

    Replaying the same RGB, calibration and pinned teacher returns the same
    source ID, including across acquisitions. The separate session label must
    never be used to manufacture a fresh image prior or be merged with a map's
    coordinate-chart label. Weight revisions are required for an honest ID.
    """
    rgb, K = np.asarray(rgb), np.asarray(K, dtype="<f8")
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Metric source identity requires native uint8 RGB")
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("Metric source identity requires finite 3x3 calibration")
    if not revision or not session_id or int(resolution) <= 0:
        raise ValueError("Metric source identity requires a pinned revision, session and resolution")
    image_hash = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
    description = dict(image_sha256=image_hash, image_shape=rgb.shape, K=K.tolist(),
                       model_id=str(model_id), revision=str(revision),
                       resolution=int(resolution), policy=DEPTH_POLICY)
    digest = hashlib.sha256(json.dumps(description, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return MetricSource(digest, image_hash, str(model_id), str(revision), int(resolution), str(session_id))


@dataclass(frozen=True)
class TranslationResponse:
    """Sparse d(world translation)/d(source log bias), at zero bias.

    Tuples make snapshots immutable, pickleable and independent of CUDA state.
    Only teacher requests materialize/copy the prefix; the fast frontend holds
    a prefix and one current anchor displacement between requests.
    """
    terms: tuple = ()  # ((source_id, (dx, dy, dz)), ...), unique, sorted

    def with_displacement(self, source_id, world_displacement):
        if not source_id:
            raise ValueError("A depth-dependent displacement needs a source ID")
        displacement = np.asarray(world_displacement, dtype=np.float64)
        if displacement.shape != (3,) or not np.isfinite(displacement).all():
            raise ValueError("Displacement must be a finite 3-vector")
        terms = dict(self.terms)
        derivative = np.asarray(terms.get(source_id, (0., 0., 0.))) - displacement
        if np.any(derivative != 0.):
            terms[source_id] = tuple(float(x) for x in derivative)
        else:
            terms.pop(source_id, None)
        return TranslationResponse(tuple(sorted(terms.items())))

    def record(self):
        return {source: list(vector) for source, vector in self.terms}

    def translation_at(self, nominal_translation, biases):
        """Frozen-correspondence response; not a refitted or corrected pose."""
        result = np.asarray(nominal_translation, dtype=np.float64).copy()
        for source, vector in self.terms:
            result += -np.expm1(-float(biases.get(source, 0.))) * np.asarray(vector)
        return result


def relative_scale_response(previous_pose, previous_response, current_pose, current_response):
    """Right-SE3-tangent Jacobians (translation first) of T_prev^-1 T_cur.

    The right-tangent rotation is R_cur.T in world coordinates. Using R_prev.T
    instead would return a translation-coordinate derivative, not the twist
    used by CROSS's conditional pose distributions. Sources cancel by identity
    before uncertainty is marginalized. Both poses use the frontend chart.
    """
    previous = dict(previous_response.terms)
    current = dict(current_response.terms)
    rotation = np.asarray(current_pose)[:3, :3]
    jacobians = {}
    for source in sorted(previous.keys() | current.keys()):
        difference = np.asarray(current.get(source, (0., 0., 0.))) - np.asarray(previous.get(source, (0., 0., 0.)))
        if np.any(difference != 0.):
            jacobians[source] = np.r_[rotation.T @ difference, np.zeros(3)].tolist()
    return jacobians


def pnp_scale_response(pose):
    """Right-tangent derivative of one fixed-correspondence metric PnP pose."""
    pose = np.asarray(pose)
    return np.r_[-pose[:3, :3].T @ pose[:3, 3], np.zeros(3)].tolist()
