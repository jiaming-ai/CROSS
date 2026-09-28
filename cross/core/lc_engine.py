import threading
import queue
from typing import Optional, Dict, Any
from loguru import logger

from cross.core.pgo import PoseGraph


class LoopClosureEngine:
    """
    Background worker that constructs and solves loop-closure pose-graphs.
    It never mutates shared state; instead, it submits solved results to an
    apply-queue consumed by the frontend thread.
    """

    def __init__(
        self,
        hypothesis_manager,
        apply_queue: 'queue.Queue',
        device: str = "cuda",
        depth: int = 1000,
        k_hop: int = 2,
        queue_size: int = 1,
    ) -> None:
        self.hm = hypothesis_manager
        self.apply_queue = apply_queue
        self.device = device
        self.depth = depth
        self.k_hop = k_hop

        # Latest-wins queue for LC jobs
        self.job_queue: 'queue.Queue' = queue.Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    def start(self):
        if self._thread is not None:
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="LoopClosureEngine", daemon=True)
        self._thread.start()
        logger.info("LoopClosureEngine started")

    def stop(self, timeout: float = 1.0):
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("LoopClosureEngine stopped")

    def submit(self, hypo_id: int, target_node_id: Optional[int] = None) -> None:
        """Submit a LC job. Latest-wins if queue is full."""
        job = {"hypo_id": hypo_id, "target_node_id": target_node_id}
        try:
            self.job_queue.put_nowait(job)
        except queue.Full:
            try:
                _ = self.job_queue.get_nowait()
            except Exception:
                pass
            self.job_queue.put_nowait(job)

    def _run(self):
        while self._running.is_set():
            try:
                job = self.job_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            hypo_id = job["hypo_id"]
            target_node_id = job.get("target_node_id")
            if target_node_id is None and self.hm.nodes:
                target_node_id = max(self.hm.nodes.keys())

            # Build pose graph under graph lock to ensure a consistent snapshot
            try:
                with self.hm.graph_lock:
                    pg = PoseGraph(
                        self.hm,
                        depth=self.depth,
                        k_hop=self.k_hop,
                        device=self.device,
                    )
                    pg.construct_for_loop_closure(
                        target_node_id=target_node_id,
                        other_hypothesis_id=hypo_id,
                    )
            except Exception as e:
                logger.exception(f"LC Engine: failed to construct pose graph: {e}")
                continue

            if len(pg.vertices) < 10 or len(pg.edges) < 10:
                logger.warning("LC Engine: pose graph too small, skipping")
                continue

            # Solve without locks
            try:
                # Fix earliest original keyframe from hypothesis 0
                original_kf_ids = [v.id for v in pg.vertices if v.id in self.hm.nodes]
                temp_vertex_ids = [v.id for v in pg.vertices if v.id not in self.hm.nodes]
                if not original_kf_ids:
                    logger.warning("LC Engine: no original keyframes found in graph")
                    continue
                fixed_node_id = min(original_kf_ids)
                optim_node_ids = set(original_kf_ids + temp_vertex_ids) - {fixed_node_id}
                pg.solve(optim_node_ids=optim_node_ids, fixed_node_ids={fixed_node_id})
            except Exception as e:
                logger.exception(f"LC Engine: PGO solve failed: {e}")
                continue

            result: Dict[str, Any] = {
                "success": True,
                "pose_graph": pg,
                "optimized_poses": pg.optimized_poses,
                "other_hypothesis_id": hypo_id,
                "target_node_id": target_node_id,
                "cost": pg.optimization_cost,
            }

            # Enqueue for application on the frontend thread
            try:
                self.apply_queue.put_nowait(result)
            except queue.Full:
                # latest-wins for apply as well
                try:
                    _ = self.apply_queue.get_nowait()
                except Exception:
                    pass
                self.apply_queue.put_nowait(result)
