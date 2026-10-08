"""Pose-graph optimisation in the background (`mapping.loop_closure.async_pgo`).

A verified loop closure optimises hypothesis 0 (GTSAM, plus the building of the graph and the posterior test).  On a
large map that takes seconds, during which the front end processes no frame.  The optimisation runs in a *forked
process* here: the child owns a copy-on-write snapshot of the whole system at the moment of the fork, runs the very
code of the synchronous path on it, and sends back the optimised poses and the edges its posterior test quarantined.
The front end keeps processing frames meanwhile and applies the result when it arrives.

Why a process: GTSAM's optimiser holds Python's GIL for the whole optimisation (measured: a spinner thread stalls for
the full duration), and the graph construction is Python.  A thread would block the front end as much as the
synchronous call.  Why a fork and not a worker with a data snapshot: the child computes the same thing as the
synchronous path on the same state (the zero-lag mode is bit-identical), and the snapshot costs a page-table copy
instead of a walk over the graph.  The child must not use CUDA (it only touches CPU state: the pose graph lives on
`state_device`, which must then be the CPU) and ends with `os._exit` so that no exit handler of the parent runs.

A result computed on the state at the fork is applied to the state of the moment it arrives:
- keyframes the optimisation moved: the optimised pose (when nothing else moved the keyframe in between; otherwise the
  correction `P_opt P_fork^-1` is left-multiplied to its present pose);
- keyframes added after the fork, and the tracked pose: the correction of the latest keyframe at the fork (they hang
  off it through the odometry chain), as a rigid transformation of the whole tail;
- edges the posterior test quarantined are quarantined here.
Loop-closure triggers that arrive while a job is in flight are remembered and re-checked against the corrected poses
when it has been applied (one job at a time).
"""
from __future__ import annotations

import os
import pickle
import select
import signal
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pypose as pp
import torch
from loguru import logger

from cross.core.pgo import as_se3

_live_children: set = set()


def _kill_children():
    for pid in list(_live_children):
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except Exception:
            pass
        _live_children.discard(pid)


import atexit
atexit.register(_kill_children)


class JobError(RuntimeError):
    pass


_zombies: list = []


def _reap(pid: int, wait: bool = False) -> int:
    """Collect a finished child's exit status without waiting for it (unless `wait`); children still exiting are
    reaped by a later call (`reap_finished`)."""
    try:
        got, status = os.waitpid(pid, 0 if wait else os.WNOHANG)
    except ChildProcessError:
        _live_children.discard(pid)
        return -1
    if got == 0:
        _zombies.append(pid)
        return -1
    _live_children.discard(pid)
    return status


def reap_finished() -> None:
    for pid in list(_zombies):
        try:
            got, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            got = pid
        if got:
            _zombies.remove(pid)
            _live_children.discard(pid)


class ForkedJob:
    """`fn()` run in a forked child; its (picklable) return value comes back through a pipe.

    `ready()` is true once the child has finished computing (it is writing its result), `result()` blocks until the
    result is read and the child reaped, `cancel()` kills the child."""

    def __init__(self, fn):
        self._fn = fn
        self.pid: Optional[int] = None
        self._fd: Optional[int] = None
        self._buf = bytearray()
        self.t_fork = 0.0              # seconds the fork call took in the parent
        self.t_start = 0.0
        self._done = False

    def start(self) -> "ForkedJob":
        r, w = os.pipe()
        t0 = time.perf_counter()
        pid = os.fork()
        if pid == 0:                                    # child: never returns
            code = 1
            try:
                os.close(r)
                _live_children.clear(); _zombies.clear()
                torch.set_num_threads(1)                 # a forked OpenMP pool does not survive
                logger.disable("cross")                  # no log lines of the child (shared sinks, buffered copies)
                try:
                    out = ("ok", self._fn())
                except BaseException:
                    out = ("error", traceback.format_exc())
                data = pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)
                with os.fdopen(w, "wb") as f:
                    f.write(data)
                code = 0
            finally:
                os._exit(code)
        os.close(w)
        self.t_fork = time.perf_counter() - t0
        self.t_start = time.perf_counter()
        self.pid, self._fd = pid, r
        _live_children.add(pid)
        return self

    def ready(self) -> bool:
        """True when the child has finished (its result, or its end, can be read)."""
        if self._done or self._fd is None:
            return True
        r, _, _ = select.select([self._fd], [], [], 0)
        return bool(r)

    def result(self, timeout: Optional[float] = None):
        """The child's return value (blocks until it has finished, at most `timeout` seconds after the fork).  Raises
        JobError when the child failed or timed out (a child that deadlocked after the fork is killed)."""
        if self._fd is None:
            raise JobError("job already collected")
        fd = self._fd
        try:
            while True:
                if timeout is not None:
                    left = timeout - (time.perf_counter() - self.t_start)
                    r, _, _ = select.select([fd], [], [], max(left, 0.0))
                    if not r:
                        os.kill(self.pid, signal.SIGKILL)
                        raise JobError(f"the worker did not finish within {timeout:.0f} s and was killed")
                chunk = os.read(fd, 1 << 22)
                if not chunk:
                    break
                self._buf += chunk
        finally:
            os.close(fd)
            self._fd = None
            status = _reap(self.pid, wait=not self._buf)    # (a finished child is reaped later: tearing down its copy of a large
            self._done = True                               # address space takes tens of ms the front end need not wait for)
        if not self._buf:
            raise JobError(f"the worker ended without a result (status {status})")
        kind, payload = pickle.loads(bytes(self._buf))
        self._buf = bytearray()
        if kind == "error":
            raise JobError(f"the worker raised:\n{payload}")
        return payload

    def cancel(self) -> None:
        if self._fd is None:
            return
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None
        try:
            os.waitpid(self.pid, 0)
        except ChildProcessError:
            pass
        _live_children.discard(self.pid)
        self._done = True


# ----------------------------------------------------------------------------------------------- the job (child side)
def _rows(hm, ids) -> np.ndarray:
    """Hypothesis-0 pose rows (float32, (n, 7)) of keyframes `ids`."""
    return np.stack([hm.nodes[i].plain_row("pose_mu", 0).detach().cpu().numpy().astype(np.float32) for i in ids]) \
        if len(ids) else np.zeros((0, 7), np.float32)


def _package(out: dict, hm, opt: dict, fork_rows: dict) -> None:
    """Optimised poses with their fork-time rows into `out` (ids / opt / fork; applied)."""
    ids = list(opt.keys())
    out["ids"] = np.asarray(ids, dtype=np.int64)
    out["opt"] = np.stack([opt[i].tensor().detach().cpu().numpy().astype(np.float32).reshape(-1) for i in ids])
    # fork-time rows; ids the first solve did not move are unchanged in this copy
    out["fork"] = np.stack([fork_rows[i] if i in fork_rows else _rows(hm, [i])[0] for i in ids])
    out["applied"] = True


def geo_pgo(hm, window_ref) -> dict:
    """The GNSS-triggered optimisation of hypothesis 0 (System._geo_maybe_optimize) on this process's copy."""
    t0 = time.perf_counter()
    out = {"success": False, "applied": False, "max_id": max(hm.nodes.keys()), "outliers": [], "message": None, "stats": {}}
    info = hm.handle_loop_closure(0, apply=False, window_ref=window_ref)
    if not info.get("success"):
        out["message"] = info.get("message")
        return out
    out["success"] = True
    pg = info["pose_graph"]
    opt = info["optimized_poses"]
    if opt:
        _package(out, hm, opt, {i: r for i, r in zip(opt.keys(), _rows(hm, list(opt.keys())))})
    out["first"] = {"cost": info.get("cost"), "initial": getattr(pg, "initial_cost", None), "lm": getattr(pg, "lm_iterations", None),
                    "window": info.get("window"), "vertices": len(getattr(pg, "vertices", [])), "factors": dict(getattr(pg, "n_factors", {})),
                    "unary": getattr(pg, "n_unary", 0)}
    out["t_compute"] = time.perf_counter() - t0
    return out


def verified_pgo(hm, verifier, lc_cfg, new_keys, window_ref) -> dict:
    """The optimisation of a verified loop closure on this process's copy of the graph: solve, posterior test of the
    step's new edges, quarantine and re-solve.  The same sequence as the synchronous path of
    `System._verified_lc_optimise` (including `test_before_apply`); returns what the parent has to apply:
    `applied` (poses to write), `ids` / `opt` / `fork` (optimised and fork-time rows), `outliers` (edges to quarantine,
    as (a, b, index in the edge's factor list, measurement)), `stats`."""
    t0 = time.perf_counter()
    max_id = max(hm.nodes.keys())
    use_post = bool(lc_cfg.use_posterior)
    remove = getattr(lc_cfg, "posterior_action", "remove") == "remove"
    defer = bool(lc_cfg.test_before_apply) and use_post and remove
    out = {"success": False, "applied": False, "max_id": max_id, "outliers": [], "message": None, "stats": {}}
    info = hm.handle_loop_closure(0, apply=False, window_ref=window_ref)
    if not info.get("success"):
        out["message"] = info.get("message")
        return out
    out["success"] = True
    pg = info["pose_graph"]
    opt = info["optimized_poses"]
    fork_rows = {i: r for i, r in zip(opt.keys(), _rows(hm, list(opt.keys())))}      # before anything is written
    stats = {"pgo": 1}
    first = {"cost": info.get("cost"), "initial": getattr(pg, "initial_cost", None), "lm": getattr(pg, "lm_iterations", None),
             "window": info.get("window"), "vertices": len(getattr(pg, "vertices", [])), "factors": dict(getattr(pg, "n_factors", {}))}
    applied = True
    if use_post:
        outliers = verifier.posterior_outliers(pg, only_keys=set(new_keys))
        if outliers:
            stats["posterior_flagged"] = len(outliers)
            if remove:
                refs = []
                bucket_of = hm.hypotheses[0].visual_edges
                for (a, b, f, c2) in outliers:
                    idx = next((k for k, g in enumerate(bucket_of[(a, b)]) if g is f), None)
                    refs.append((a, b, idx, np.array(f.mean_np, dtype=np.float64), float(c2)))
                out["outliers"] = refs
                if defer and len(outliers) == len(new_keys):
                    applied = False                          # every new edge was an outlier: the graph stays as it is
                else:
                    for (a, b, f, c2) in outliers:
                        verifier.remove_edge(a, b, f)
                    if not defer:
                        hm._write_poses(opt)                 # the first solution was applied before its test
                    info2 = hm.handle_loop_closure(0, apply=False, window_ref=window_ref)
                    if info2.get("success"):
                        # an applied first solution moved its keyframes even where the second graph dropped them
                        opt = {**opt, **info2["optimized_poses"]} if not defer else info2["optimized_poses"]
                        pg = info2["pose_graph"]
                        out["second"] = {"cost": info2.get("cost"), "vertices": len(getattr(pg, "vertices", []))}
                    elif defer:
                        opt = {}                              # the re-solve failed and the first solution was never applied
    if applied and opt:
        _package(out, hm, opt, fork_rows)
    out["stats"] = stats
    out["first"] = first
    out["t_compute"] = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------------------------- the manager (parent side)
class AsyncPgo:
    """One background optimisation at a time for a System.  See the module docstring."""

    def __init__(self, system, lag_steps: int = -1, min_job_s: float = 0.1):
        self.system = system
        self.lag = int(lag_steps)         # -1: apply when finished; k >= 0: apply exactly k steps after the submission
        # Free-running mode: an optimisation expected to take less than max(min_job_s, 2 x the front end's cost of a
        # background job) is run in the front end as before (it would not stall it, and a small map stays bit-identical to the
        # synchronous path).  The expectation is the running mean of the optimisations seen so far (synchronous or not).
        self.min_job_s = float(min_job_s)
        self.timeout_s = 900.0            # a worker that has not finished by then is killed (and the job re-queued)
        self.est_job_s = 0.0
        self.est_overhead_s = 0.03
        self.job: Optional[ForkedJob] = None
        self.job_meta: dict = {}
        self.pending: Dict[Tuple[int, int], None] = {}       # loop triggers that arrived while a job was in flight
        self.failures = 0
        self.disabled = False
        self.stats = {"submitted": 0, "applied": 0, "discarded": 0, "failed": 0, "fork_s": 0.0, "apply_s": 0.0,
                      "compute_s": 0.0, "wait_s": 0.0, "stale_steps": 0, "resubmitted": 0}

    def worth_it(self) -> bool:
        """Should the next optimisation go to the background?  (Always in the reproducible modes, lag >= 0.)"""
        return self.lag >= 0 or self.est_job_s > max(self.min_job_s, 2.0 * self.est_overhead_s)

    def observe(self, job_s: float, overhead_s: Optional[float] = None) -> None:
        """Duration of an optimisation (its compute time) and, for a background one, what it cost the front end."""
        self.est_job_s = job_s if self.est_job_s == 0.0 else 0.5 * self.est_job_s + 0.5 * job_s
        if overhead_s is not None:
            self.est_overhead_s = 0.5 * self.est_overhead_s + 0.5 * overhead_s

    @property
    def busy(self) -> bool:
        return self.job is not None

    def usable(self) -> bool:
        """The fork runs the CPU code of the optimisation on a copy of the state: not with the state on a GPU, not
        for coordinate-chart / conditional graphs (their write-back needs the optimiser's own graph)."""
        sys_ = self.system
        hm = sys_.hypothesis_manager
        return (not self.disabled and hasattr(os, "fork") and str(getattr(hm, "device", "cpu")).startswith("cpu")
                and not hm.chart_aware and hm.source_states is None)

    # ---- submit
    def submit(self, new_keys, window_ref, step: int, kind: str = "verified", **extra) -> None:
        reap_finished()
        """Fork the optimisation: kind "verified" (a verified loop closure for the edges `new_keys`) or "geo" (the
        GNSS-triggered one; `extra` is kept in the job's meta for the application)."""
        sys_ = self.system
        hm = sys_.hypothesis_manager
        v = sys_._lc_verifier
        lc_cfg = sys_.config.mapping.loop_closure
        keys = list(new_keys)
        self.job_meta = {"step": step, "keys": keys, "window_ref": window_ref, "pose_epoch": hm.pose_epoch, "kind": kind,
                         "graph_epoch": hm.graph_epoch, "n_nodes": len(hm.nodes), "t_submit": time.perf_counter(), **extra}
        if kind == "geo":
            self.job = ForkedJob(lambda: geo_pgo(hm, window_ref)).start()
        else:
            self.job = ForkedJob(lambda: verified_pgo(hm, v, lc_cfg, keys, window_ref)).start()
        self.job_meta["fork_s"] = self.job.t_fork
        self.stats["submitted"] += 1
        self.stats["fork_s"] += self.job.t_fork

    def note_pending(self, new_keys) -> None:
        for k in new_keys:
            self.pending[k] = None

    # ---- collect
    def due(self, step: int) -> bool:
        if self.job is None:
            return False
        if self.lag >= 0:
            return step >= self.job_meta["step"] + self.lag
        return self.job.ready()

    def collect(self, step: int):
        """The finished job's result (blocking when it is due by `lag`) or None when it failed."""
        job, meta = self.job, self.job_meta
        t0 = time.perf_counter()
        try:
            res = job.result(timeout=self.timeout_s)
        except JobError as ex:
            self.job = None
            self.failures += 1
            self.stats["failed"] += 1
            logger.warning(f"background pose-graph optimisation failed ({str(ex).strip().splitlines()[-1]}); keys re-queued")
            if self.failures >= 3:
                self.disabled = True
                logger.warning("background pose-graph optimisation switched off after 3 failures (synchronous from here)")
            self.note_pending(meta["keys"])
            return None, meta
        meta["wait_s"] = time.perf_counter() - t0               # the front end waiting for the worker (lag mode)
        self.stats["wait_s"] += meta["wait_s"]
        self.job = None
        self.failures = 0
        return res, meta

    def cancel(self, requeue: bool = True) -> None:
        if self.job is not None:
            self.job.cancel()
            self.job = None
            self.stats["discarded"] += 1
            if requeue:
                self.note_pending(self.job_meta.get("keys", []))
