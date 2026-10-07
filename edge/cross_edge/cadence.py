"""The CROSS back end's observation cadence, run on the edge's own odometry: which frames' images the server needs."""

from types import SimpleNamespace

import numpy as np

# the back end's settings the cadence copy needs (cross.core.config.PoseEstConfig); the server sends them when a session
# opens (cadence_params), so the edge needs no CROSS configuration
CADENCE_KEYS = ("obs_min_translation", "obs_min_rotation", "obs_max_interval_steps", "obs_warmup_steps",
                "obs_confident_max_interval_steps", "obs_confident_min_translation", "obs_confident_min_rotation")


def cadence_params(pose_est_cfg) -> dict:
    """The cadence settings of a back end's configuration (an object with the CADENCE_KEYS attributes) as a dict."""
    return {k: getattr(pose_est_cfg, k) for k in CADENCE_KEYS if hasattr(pose_est_cfg, k)}


def cadence_config(params: dict):
    """A configuration object for ObservationCadence from cadence_params' dict."""
    return SimpleNamespace(**params)


class ObservationCadence:
    """The back end's observation cadence (cross.core.system.System._should_skip_observation: every frame for
    obs_warmup_steps frames after an initialization, then after obs_min_translation m, obs_min_rotation rad or
    obs_max_interval_steps mapped frames since the last observation) on the edge's own odometry: the frames whose images
    the server needs.  The back end decides on the same motion (the deltas the edge sends), so the two agree up to
    rounding; a frame the back end wants without its image is observed at the next one that has it.  Its adaptive
    relaxation while one hypothesis dominates (obs_confident_*) follows the server's last reply (confident): exact
    without latency; with latency the edge sends a few images the back end skips, or the back end observes a frame later."""

    def __init__(self, pose_est_cfg):
        c = self.cfg = pose_est_cfg
        self.min_t, self.min_r = float(c.obs_min_translation), float(c.obs_min_rotation)
        self.max_steps, self.warmup = int(c.obs_max_interval_steps), int(c.obs_warmup_steps)
        self.every_frame = self.min_t <= 0 and self.min_r <= 0 and self.max_steps <= 1
        relax = int(getattr(c, "obs_confident_max_interval_steps", 0))
        self.relaxed = (relax, float(c.obs_confident_min_translation), float(c.obs_confident_min_rotation)) if relax > 0 else None
        self.confident = False                   # the back end's state in its last reply (MapServer: map.confident)
        self.processed = self.start = self.steps = 0
        self.T = np.eye(4)
        self.missing = False

    def sync(self, state):
        """Continue from the back end's actual state (MapServer.cadence_state, after a map load)."""
        self.processed, self.start = int(state["processed"]), int(state["session_start"])
        self.steps = int(state["steps_since_obs"])
        T = state.get("T_since_obs")
        self.T = np.eye(4) if T is None else np.asarray(T, dtype=np.float64)
        self.unknown = T is None and self.processed > 0       # its motion since the observation is not known: observe
        self.missing = bool(state.get("kidnap"))

    def frame(self, delta, map_frame: bool) -> bool:
        """The motion of this frame (None: missing) and whether the back end steps it; True if it will observe it (or
        initialize on it)."""
        self._undo = None
        if delta is None:
            self.missing = True                  # the back end re-initializes at its next step (kidnapped)
        else:
            self.T = self.T @ np.asarray(delta, dtype=np.float64)
        if not map_frame:
            return False
        self.processed += 1
        if self.processed == 1 or self.missing:
            self.missing = False
            self.start, self.steps, self.T = self.processed, 0, np.eye(4)
            return True                          # (the image is needed: no veto)
        if self.every_frame:
            return True
        self.steps += 1
        unknown = getattr(self, "unknown", False)
        max_steps, min_t, min_r = self.max_steps, self.min_t, self.min_r
        if self.relaxed is not None and self.confident:
            max_steps, min_t, min_r = self.relaxed
        if self.processed - self.start <= self.warmup or self.steps >= max_steps or unknown:
            observe = True
        else:
            moved = min_t > 0 and float(np.linalg.norm(self.T[:3, 3])) >= min_t
            angle = float(np.arccos(np.clip((np.trace(self.T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
            observe = moved or (min_r > 0 and angle >= min_r)
        if observe:
            self._undo = (self.steps, self.T, unknown)
            self.steps, self.T, self.unknown = 0, np.eye(4), False
        return observe

    def can_veto(self) -> bool:
        return self._undo is not None

    def veto(self):
        """This frame's image is not sent after all (the rate cap): the back end defers the observation to the next
        frame with an image, and the copy keeps counting as it does."""
        if self._undo is not None:
            self.steps, self.T, self.unknown = self._undo
            self._undo = None
