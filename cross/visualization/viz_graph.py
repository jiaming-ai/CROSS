

from typing import List, Optional, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from loguru import logger

from cross.core.pgo import PoseGraph
from cross.core.types import EdgeType


def _get_subgraph_nodes(
    pose_graph: PoseGraph,
    last_k_nodes: int,
) -> Set[int]:
    """
    Get a subset of nodes for visualization: last k nodes + their first-order neighbors.

    Args:
        pose_graph: PoseGraph object
        last_k_nodes: Number of most recent nodes to include

    Returns:
        Set of node IDs (kf_ids) to visualize
    """
    if last_k_nodes <= 0 or last_k_nodes >= len(pose_graph.vertices):
        # Return all nodes if k is invalid or larger than graph
        return {v.id for v in pose_graph.vertices}

    # Get last k vertices
    last_k_vertices = pose_graph.vertices.copy()
    # sort by kf_id
    last_k_vertices.sort(key=lambda x: x.id)
    last_k_vertices = last_k_vertices[-last_k_nodes:]

    subset_ids = {v.id for v in last_k_vertices}

    # Find first-order neighbors
    neighbor_ids = set()
    for (u, v, _) in pose_graph.edges:
        if u in subset_ids:
            neighbor_ids.add(v)
        if v in subset_ids:
            neighbor_ids.add(u)

    # Combine subset and neighbors
    result = subset_ids | neighbor_ids
    logger.info(
        f"Subgraph: {last_k_nodes} core nodes + {len(neighbor_ids - subset_ids)} "
        f"neighbors = {len(result)} total nodes"
    )

    return result


def visualize_pose_graph(
    pose_graph: PoseGraph,
    title: str = "Pose Graph",
    node_size: int = 500,
    font_size: int = 8,
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None,
    last_k_nodes: Optional[int] = None,
    show_interactive: bool = False,
) -> None:
    """
    Visualize a pose graph using networkx with matplotlib.

    Features:
    - Interactive plot (pan, zoom)
    - Node labels: V_id:KF_id:Comp_id
    - Edge colors by type (odometry, visual, loop_closure)
    - Edge labels showing std magnitude

    Args:
        pose_graph: PoseGraph object with constructed vertices and edges
        title: Plot title
        node_size: Size of node markers
        font_size: Font size for labels
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        last_k_nodes: If specified, visualize only the last k nodes and their
                      first-order neighbors. Useful for large graphs. Set to None
                      to visualize all nodes.
    """

    # Determine which nodes to visualize
    if last_k_nodes is not None:
        visible_nodes = _get_subgraph_nodes(pose_graph, last_k_nodes)
    else:
        visible_nodes = {v.id for v in pose_graph.vertices}

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with attributes (only visible ones)
    for vertex in pose_graph.vertices:
        if vertex.id not in visible_nodes:
            continue
        G.add_node(
            vertex.id,
            comp_id=vertex.original_comp_id,
            kf_id=vertex.original_kf_id,
            temporary=vertex.temporary,
            label=f"V{vertex.id}:KF{vertex.original_kf_id}:C{vertex.original_comp_id}",
        )

    # Color map for edge types
    edge_color_map = {
        EdgeType.ODOMETRY: "blue",
        EdgeType.VISUAL: "green",
        EdgeType.LOOP_CLOSURE: "red",
        EdgeType.CHAIN: "orange",
        EdgeType.PROXIMITY: "purple",
    }

    # Add edges with attributes (only between visible nodes)
    edge_data = []
    for (u, v, factors) in pose_graph.edges:
        # Skip edges that connect to nodes outside visible set
        if u not in visible_nodes or v not in visible_nodes:
            continue

        assert u in G.nodes and v in G.nodes, f"Node {u} or {v} not found in graph"

        # Process each factor (there might be multiple factors per edge)
        for factor in factors:
            # Compute std magnitude (L2 norm of std vector)
            std_magnitude = np.linalg.norm(
                factor.std.tensor().cpu().numpy().flatten()
            )

            edge_type = getattr(factor, 'type', 'unknown')
            assert edge_type in edge_color_map, f"Edge type {edge_type} not found in edge_color_map"
            color = edge_color_map.get(edge_type)

            G.add_edge(u, v)
            edge_data.append({
                'u': u,
                'v': v,
                'type': edge_type,
                'color': color,
                'std': std_magnitude,
            })

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Use spring layout for better visualization
    # For larger graphs, you might want to use other layouts
    try:
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)
    except Exception as e:
        logger.warning(f"Spring layout failed: {e}, using shell layout instead")
        pos = nx.shell_layout(G)

    # Draw nodes with different colors based on temporary status
    # Create color list based on temporary attribute
    node_colors = []
    for node_id in G.nodes():
        is_temporary = G.nodes[node_id]['temporary']
        node_colors.append('orange' if is_temporary else 'lightblue')

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_colors,
        edgecolors='black',
        linewidths=1.5,
        ax=ax,
    )

    # Draw node labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=font_size,
        font_weight='bold',
        ax=ax,
    )

    # Group edges by type and draw them separately with different colors
    edge_types = {}
    for edge in edge_data:
        edge_type = edge['type']
        if edge_type not in edge_types:
            edge_types[edge_type] = []
        edge_types[edge_type].append(edge)

    # Draw edges by type with varying thickness based on std (weight)
    # Collect all std values to normalize
    all_stds = [e['std'] for e in edge_data]
    min_std = min(all_stds) if all_stds else 1.0
    max_std = max(all_stds) if all_stds else 1.0
    std_range = max_std - min_std if max_std > min_std else 1.0

    for edge_type, edges in edge_types.items():
        color = edge_color_map.get(edge_type, "gray")

        # Draw each edge with its own width based on std
        # Lower std (more certain) -> thicker line
        # Higher std (less certain) -> thinner line
        for edge in edges:
            # Normalize std to [0, 1] range, then invert and map to width range [0.5, 3.0]
            normalized_std = (edge['std'] - min_std) / std_range if std_range > 0 else 0.5
            width = 3.0 - (normalized_std * 2.5)  # Inverted: lower std = thicker

            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(edge['u'], edge['v'])],
                edge_color=color,
                width=width,
                alpha=0.6,
                arrows=True,
                arrowsize=15,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1',
                ax=ax,
                label=edge_type if edge == edges[0] else "",  # Only label once per type
            )

    # Add edge labels (std magnitude)
    # Only show labels for a subset to avoid clutter
    edge_labels = {}
    for edge in edge_data:
        key = (edge['u'], edge['v'])
        if key not in edge_labels:  # Only show first factor if multiple
            edge_labels[key] = f"{edge['std']:.2e}"

    # Only show edge labels if graph is not too large
    if len(edge_labels) < 500:
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=max(6, font_size - 2),
            ax=ax,
        )

    # Add legend for edge types
    ax.legend(loc='upper right', fontsize=10)

    # Count temporary vs permanent nodes
    temp_count = sum(1 for n in G.nodes() if G.nodes[n]['temporary'])
    perm_count = len(G.nodes) - temp_count

    # Add graph statistics as text
    stats_text = (
        f"Nodes: {len(G.nodes)} ({perm_count} permanent, {temp_count} temporary)\n"
        f"Edges: {len(edge_data)}\n"
        f"Odometry: {len(edge_types.get('odometry', []))}\n"
        f"Visual: {len(edge_types.get('visual', []))}\n"
        f"Loop Closure: {len(edge_types.get('loop_closure', []))}\n"
        f"\nNode colors: blue=permanent, orange=temporary"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.axis('off')
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Pose graph visualization saved to {save_path}")

    # Show with interactive controls
    if show_interactive:
        plt.show(block=False)
        logger.info("Pose graph visualization displayed. Use matplotlib controls to pan/zoom.")


def visualize_pose_graph_3d(
    pose_graph: PoseGraph,
    title: str = "Pose Graph (3D Poses)",
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None,
    last_k_nodes: Optional[int] = None,
) -> None:
    """
    Visualize pose graph with 3D trajectory overlay.

    Shows both the graph structure and the actual spatial positions of poses.

    Args:
        pose_graph: PoseGraph object with constructed vertices and edges
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        last_k_nodes: If specified, visualize only the last k nodes and their
                      first-order neighbors. Useful for large graphs. Set to None
                      to visualize all nodes.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection on older matplotlib)

    from cross.core.pgo import PoseGraph as PG

    if not isinstance(pose_graph, PG):
        logger.error("Input must be a PoseGraph object")
        return

    if not pose_graph.vertices:
        logger.warning("Pose graph is empty. No visualization to show.")
        return

    # Determine which nodes to visualize
    if last_k_nodes is not None:
        visible_nodes = _get_subgraph_nodes(pose_graph, last_k_nodes)
    else:
        visible_nodes = {v.id for v in pose_graph.vertices}

    # Extract 3D positions from poses (only visible ones)
    positions = {}
    for vertex in pose_graph.vertices:
        if vertex.id not in visible_nodes:
            continue
        pose_tensor = vertex.pose.tensor().cpu().numpy()
        # pypose SE3: [x, y, z, qx, qy, qz, qw]
        positions[vertex.id] = pose_tensor[:3]

    # Create figure with 3D subplot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Color map for edge types
    edge_color_map = {
        EdgeType.ODOMETRY: "blue",
        EdgeType.VISUAL: "green",
        EdgeType.LOOP_CLOSURE: "red",
        EdgeType.CHAIN: "orange",
        EdgeType.PROXIMITY: "purple",
    }

    # Plot edges in 3D (only between visible nodes)
    for (u, v, factors) in pose_graph.edges:
        # Skip edges that connect to nodes outside visible set
        if u not in visible_nodes or v not in visible_nodes:
            continue
        if u not in positions or v not in positions:
            continue

        for factor in factors:
            edge_type = getattr(factor, 'type', 'unknown')
            color = edge_color_map.get(edge_type, "gray")

            pos_u = positions[u]
            pos_v = positions[v]

            ax.plot(
                [pos_u[0], pos_v[0]],
                [pos_u[1], pos_v[1]],
                [pos_u[2], pos_v[2]],
                color=color,
                alpha=0.6,
                linewidth=1.5,
            )

    # Plot nodes with different colors based on temporary status
    node_positions = np.array(list(positions.values()))
    node_colors = []
    for vertex_id in positions.keys():
        vertex = pose_graph.vertex_map[vertex_id]
        node_colors.append('orange' if vertex.temporary else 'lightblue')

    ax.scatter(
        node_positions[:, 0],
        node_positions[:, 1],
        node_positions[:, 2],
        c=node_colors,
        s=50,
        edgecolors='black',
        linewidths=1.0,
        marker='o',
    )

    # Add labels for key nodes (first, last, and some intermediate)
    num_nodes = len(positions)
    label_indices = [0, num_nodes // 2, num_nodes - 1] if num_nodes > 2 else range(num_nodes)

    for i, (vertex_id, pos) in enumerate(positions.items()):
        if i in label_indices:
            vertex = pose_graph.vertex_map[vertex_id]
            label = f"V{vertex_id}:KF{vertex.original_kf_id}:C{vertex.original_comp_id}"
            ax.text(pos[0], pos[1], pos[2], label, fontsize=8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # Add legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Odometry'),
        Line2D([0], [0], color='green', lw=2, label='Visual'),
        Line2D([0], [0], color='red', lw=2, label='Loop Closure'),
        Patch(facecolor='lightblue', edgecolor='black', label='Permanent Node'),
        Patch(facecolor='orange', edgecolor='black', label='Temporary Node'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"3D pose graph visualization saved to {save_path}")

    # Show with interactive controls
    plt.show(block=False)
    logger.info("3D pose graph visualization displayed. Use matplotlib controls to rotate/zoom.")


def visualize_sparse_graph_matplotlib(
    system,
    path: Optional[List[int]] = None,
    title: str = "Sparse Graph",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
    edge_types: Optional[List[str]] = None,
    show_interactive: bool = True,
    permanent_only: bool = False,
    kf_id_min: Optional[int] = None,
    kf_id_max: Optional[int] = None,
) -> None:
    """
    Visualize the sparse graph using matplotlib with networkx layout.

    Shows:
    - All keyframes as nodes (permanent vs temporary, path-highlighted if provided)
    - Odometry edges (blue solid lines)
    - Backbone edges (green solid lines) - consecutive permanent keyframes
    - Chain shortcuts (orange dashed lines) - long-range chain shortcuts
    - Optional planned path highlighted (red thick lines)

    Args:
        system: System object containing hypothesis_manager and sparse_graph
        path: Optional list of keyframe IDs representing a planned path to highlight
        title: Plot title
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
        edge_types: List of edge types to show. Options: ['odometry', 'backbone', 'chain', 'path'].
                   Default is ['backbone', 'chain'] to show sparse graph structure.
        show_interactive: Whether to show the plot interactively
        node_size: Size of node markers
        font_size: Font size for labels
        permanent_only: If True, only show permanent keyframes
        kf_id_min: Minimum keyframe ID to show (inclusive), None for no limit
        kf_id_max: Maximum keyframe ID to show (inclusive), None for no limit
    """
    hypothesis_manager = system.hypothesis_manager
    # Get sparse_graph from planning_system if available (new architecture)
    if hasattr(system, 'planning_system') and system.planning_system is not None:
        sparse_graph = system.planning_system.sparse_graph
    else:
        # Fallback for backward compatibility
        sparse_graph = system.sparse_graph

    # Default to showing backbone and chain edges for clarity
    if edge_types is None:
        edge_types = ['backbone', 'chain']

    # Get permanent keyframe IDs
    permanent_kf_ids = set(sparse_graph.permanent_kf_ids)

    # Determine which keyframes to include
    if permanent_only:
        # Only show permanent keyframes
        all_kf_ids = permanent_kf_ids.copy()
    else:
        # Get all keyframes from hypothesis manager
        all_kf_ids = set()

        # Collect all keyframe IDs from odometry edges
        for (id1, id2) in hypothesis_manager.odom_edges.keys():
            all_kf_ids.add(id1)
            all_kf_ids.add(id2)

        # Collect keyframe IDs from backbone and shortcut edges
        for (id1, id2) in sparse_graph.backbone_edges.keys():
            all_kf_ids.add(id1)
            all_kf_ids.add(id2)
        for (id1, id2) in sparse_graph.shortcut_edges.keys():
            all_kf_ids.add(id1)
            all_kf_ids.add(id2)

    # Apply keyframe ID range filter
    if kf_id_min is not None or kf_id_max is not None:
        filtered_kf_ids = set()
        for kf_id in all_kf_ids:
            if kf_id_min is not None and kf_id < kf_id_min:
                continue
            if kf_id_max is not None and kf_id > kf_id_max:
                continue
            filtered_kf_ids.add(kf_id)
        all_kf_ids = filtered_kf_ids

    logger.info(f"Visualizing sparse graph (matplotlib) with {len(all_kf_ids)} keyframes "
                f"(permanent_only={permanent_only}, range=[{kf_id_min}, {kf_id_max}])")

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with attributes (filter by permanent_only if needed)
    for kf_id in all_kf_ids:
        if kf_id in hypothesis_manager.nodes:
            is_permanent = kf_id in permanent_kf_ids
            # Skip temporary nodes if permanent_only is True
            if permanent_only and not is_permanent:
                continue
            G.add_node(
                kf_id,
                permanent=is_permanent,
                label=f"KF{kf_id}",
            )

    # Prepare edge data by type
    edge_data = {
        'odometry': [],
        'backbone': [],
        'chain': [],
        'path': []
    }

    # Add odometry edges
    if 'odometry' in edge_types:
        for (id1, id2) in hypothesis_manager.odom_edges.keys():
            if id1 in G.nodes and id2 in G.nodes:
                G.add_edge(id1, id2)
                edge_data['odometry'].append((id1, id2))

    # Add backbone edges (consecutive permanent keyframes)
    if 'backbone' in edge_types:
        for (id1, id2) in sparse_graph.backbone_edges.keys():
            if id1 in G.nodes and id2 in G.nodes:
                G.add_edge(id1, id2)
                edge_data['backbone'].append((id1, id2))

    # Add chain shortcut edges (long-range shortcuts)
    if 'chain' in edge_types:
        for (id1, id2) in sparse_graph.shortcut_edges.keys():
            if id1 in G.nodes and id2 in G.nodes:
                G.add_edge(id1, id2)
                edge_data['chain'].append((id1, id2))

    # Prepare path edges
    path_set = set(path) if path else set()
    if 'path' in edge_types and path:
        for i in range(len(path) - 1):
            if path[i] in G.nodes and path[i+1] in G.nodes:
                edge_data['path'].append((path[i], path[i+1]))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Use spring layout for better visualization
    try:
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)
    except Exception as e:
        logger.warning(f"Spring layout failed: {e}, using shell layout instead")
        pos = nx.shell_layout(G)

    # Draw odometry edges (blue solid lines)
    if edge_data['odometry']:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edge_data['odometry'],
            edge_color='blue',
            width=1.5,
            alpha=0.6,
            arrows=True,
            arrowsize=10,
            arrowstyle='->',
            ax=ax,
            label='Odometry',
        )

    # Draw backbone edges (green solid lines)
    if edge_data['backbone']:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edge_data['backbone'],
            edge_color='green',
            width=2,
            alpha=0.8,
            arrows=True,
            arrowsize=10,
            arrowstyle='->',
            ax=ax,
            label='Backbone',
        )

    # Draw chain shortcut edges (orange dashed lines)
    if edge_data['chain']:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edge_data['chain'],
            edge_color='orange',
            width=2,
            alpha=0.7,
            style='dashed',
            arrows=True,
            arrowsize=10,
            arrowstyle='->',
            ax=ax,
            label='Chain Shortcut',
        )

    # Draw path edges (red thick lines)
    if edge_data['path']:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edge_data['path'],
            edge_color='red',
            width=4,
            alpha=0.9,
            arrows=True,
            arrowsize=15,
            arrowstyle='->',
            ax=ax,
            label='Path',
        )

    # Create node color list based on permanent status and path membership
    node_colors = []
    node_sizes = []
    for node_id in G.nodes():
        is_permanent = G.nodes[node_id]['permanent']
        in_path = node_id in path_set

        if is_permanent and in_path:
            node_colors.append('red')
            node_sizes.append(500 * 1.5)
        elif is_permanent:
            node_colors.append('lightblue')
            node_sizes.append(500)
        elif in_path:
            node_colors.append('orange')
            node_sizes.append(500 * 1.2)
        else:
            node_colors.append('lightgray')
            node_sizes.append(500 * 0.8)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='black',
        linewidths=1.5,
        ax=ax,
    )

    # Draw node labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=8,
        font_weight='bold',
        ax=ax,
    )

    # Add legend
    ax.legend(loc='upper right', fontsize=10)

    # Add statistics text
    stats = sparse_graph.get_stats()
    perm_count = sum(1 for n in G.nodes() if G.nodes[n]['permanent'])
    temp_count = len(G.nodes) - perm_count

    stats_text = (
        f"Sparse Graph Stats:\n"
        f"Total KFs: {len(G.nodes)}\n"
        f"Permanent: {perm_count}\n"
        f"Temporary: {temp_count}\n"
        f"Backbone Edges: {stats['backbone_edges']}\n"
        f"Chain Shortcuts: {stats['shortcut_edges']}\n"
        f"Max Stride: {stats['max_stride']}\n"
        f"\nEdge types shown:\n"
        f"{', '.join(edge_types)}\n"
        f"\nNode colors:\n"
        f"red=path perm, orange=path temp\n"
        f"blue=perm, gray=temp"
    )

    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.axis('off')
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Sparse graph visualization saved to {save_path}")

    # Show with interactive controls
    if show_interactive:
        plt.show(block=False)
        logger.info("Sparse graph visualization displayed. Use matplotlib controls to pan/zoom.")

