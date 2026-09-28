"""Coordinate provenance for CROSS's existing pose-mixture message.

A chart is a coordinate frame, not an association decision or a GMM slot.
These labels prevent comparisons across unconnected gauges. They introduce
no new geometric evidence, scale estimate, or commitment threshold.
"""
import numpy as np
import torch
from sklearn.cluster import DBSCAN


def cluster_by_chart(projected, charts, eps, min_samples):
    """Apply the inherited clustering independently in each known chart."""
    charts = np.asarray(charts)
    if len(charts) != len(projected) or np.any(charts < 0):
        raise ValueError("Every active proposal needs a known coordinate chart")
    labels = np.full(len(charts), -1, dtype=np.int64)
    offset = 0
    # Preserve the legacy order within each chart and deterministic birth order.
    for chart in dict.fromkeys(charts.tolist()):
        indices = np.flatnonzero(charts == chart)
        local = DBSCAN(eps=eps, min_samples=min_samples).fit(projected[indices]).labels_
        valid = local >= 0
        labels[indices[valid]] = local[valid] + offset
        if valid.any():
            offset += int(local[valid].max()) + 1
    return labels


def restore_node_charts(nodes, odom_edges, visual_edges):
    """Restore committed charts, inferring legacy labels from graph connectivity.

    Only committed component-0 edges establish a common frame. Acquisition
    atlas, numerical proximity, and uncommitted alternatives never do. Like
    the inherited map loader, tracking hypotheses restart after loading.
    """
    if not nodes:
        return 0
    committed = list(odom_edges)
    committed += [key for key, edges in visual_edges.items()
                  if any(e.from_comp_id == e.to_comp_id == 0 for e in edges)]
    present = [getattr(kf, "pose_charts", None) is not None for kf in nodes.values()]
    if any(present) and not all(present):
        raise ValueError("Map mixes labeled and unlabeled coordinate charts")
    if not any(present):
        parent = {i: i for i in nodes}

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b in committed:
            if a in nodes and b in nodes:
                a, b = root(a), root(b)
                parent[max(a, b)] = min(a, b)
        labels = {r: i for i, r in enumerate(sorted({root(i) for i in nodes}))}
        for i, kf in nodes.items():
            kf.pose_charts = torch.full(kf.pose_weights.shape, -1, dtype=torch.long,
                                       device=kf.pose_weights.device)
            kf.pose_charts[0] = labels[root(i)]
    for kf in nodes.values():
        if kf.pose_charts.shape != kf.pose_weights.shape or int(kf.pose_charts[0]) < 0:
            raise ValueError("Invalid committed chart provenance")
        # Only graph hypothesis 0 is persisted by CROSS. Do not revive saved
        # alternatives as retrieval evidence after their branch was discarded.
        kf.pose_weights[1:] = 0
        kf.pose_charts[1:] = -1
    for a, b in committed:
        if a in nodes and b in nodes and int(nodes[a].pose_charts[0]) != int(nodes[b].pose_charts[0]):
            raise ValueError("Committed graph edge crosses unmerged coordinate charts")
    return 1 + max(int(kf.pose_charts[0]) for kf in nodes.values())
