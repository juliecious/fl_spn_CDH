"""
Causal Graph Visualization Utilities.

Provides functions to visualize DAGs (Directed Acyclic Graphs) for causal discovery experiments.
Supports:
- Ground truth vs predicted graph comparison
- Multiple layout algorithms (hierarchical, circular, spring, graphviz)
- Difference highlighting (TP, FP, FN edges)
- Export to various formats (PNG, PDF, SVG)
- Integration with experiment pipelines

Dependencies: networkx, matplotlib, pygraphviz (optional, for better layouts)
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Optional, Tuple, Dict
import os


# ------------------------------------------------------------------------------
# Core Visualization Functions
# ------------------------------------------------------------------------------


def adjacency_to_networkx(
    adj_matrix: np.ndarray, node_names: Optional[List[str]] = None
) -> nx.DiGraph:
    """
    Convert adjacency matrix to NetworkX directed graph.

    Args:
        adj_matrix: Binary adjacency matrix [n, n] where adj[i,j]=1 means i->j
        node_names: List of node names (default: G0, G1, ...)

    Returns:
        NetworkX DiGraph
    """
    n_nodes = adj_matrix.shape[0]

    if node_names is None:
        node_names = [f"G{i}" for i in range(n_nodes)]

    G = nx.DiGraph()
    G.add_nodes_from(node_names)

    # Add edges
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i, j] == 1:
                G.add_edge(node_names[i], node_names[j])

    return G


def get_layout(G: nx.DiGraph, layout: str = "hierarchical") -> Dict:
    """
    Get node positions for graph layout.

    Args:
        G: NetworkX graph
        layout: Layout algorithm:
            - "hierarchical": Top-down layered (best for DAGs)
            - "circular": Nodes in a circle
            - "spring": Force-directed layout
            - "shell": Concentric circles
            - "graphviz": Graphviz dot layout (requires pygraphviz)

    Returns:
        Dictionary mapping node -> (x, y) position
    """
    if layout == "hierarchical":
        return _hierarchical_layout(G)
    elif layout == "circular":
        return nx.circular_layout(G)
    elif layout == "spring":
        return nx.spring_layout(G, seed=42, k=2, iterations=50)
    elif layout == "shell":
        return nx.shell_layout(G)
    elif layout == "graphviz":
        try:
            return nx.nx_agraph.graphviz_layout(G, prog="dot")
        except ImportError:
            print(
                "Warning: pygraphviz not installed, falling back to hierarchical layout"
            )
            return _hierarchical_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")


def _hierarchical_layout(G: nx.DiGraph) -> Dict:
    """
    Create hierarchical (top-down) layout for DAG.
    Nodes at the same depth are placed at the same height.
    """
    # Compute node depths (topological generations)
    try:
        depths = {}
        for node in nx.topological_sort(G):
            # Node depth = max(parent depths) + 1
            predecessors = list(G.predecessors(node))
            if not predecessors:
                depths[node] = 0
            else:
                depths[node] = max(depths[p] for p in predecessors) + 1
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        # Not a DAG (contains cycles), fall back to spring layout
        print(
            "Warning: Graph contains cycles, using spring layout instead of hierarchical"
        )
        return nx.spring_layout(G, seed=42, k=2, iterations=50)

    # Group nodes by depth
    max_depth = max(depths.values())
    levels = [[] for _ in range(max_depth + 1)]
    for node, depth in depths.items():
        levels[depth].append(node)

    # Assign positions
    pos = {}
    for depth, nodes in enumerate(levels):
        n_nodes = len(nodes)
        y = 1.0 - (depth / max(max_depth, 1))  # Top to bottom

        for i, node in enumerate(nodes):
            if n_nodes == 1:
                x = 0.5
            else:
                x = i / (n_nodes - 1)
            pos[node] = (x, y)

    return pos


def visualize_dag(
    adj_matrix: np.ndarray,
    node_names: Optional[List[str]] = None,
    layout: str = "hierarchical",
    title: str = "Causal Graph",
    figsize: Tuple[int, int] = (10, 8),
    node_size: int = 2000,
    font_size: int = 12,
    edge_width: float = 2.0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize a single causal DAG.

    Args:
        adj_matrix: Binary adjacency matrix
        node_names: Node labels
        layout: Layout algorithm
        title: Plot title
        figsize: Figure size
        node_size: Node circle size
        font_size: Label font size
        edge_width: Edge line width
        save_path: If provided, save figure to this path
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    G = adjacency_to_networkx(adj_matrix, node_names)
    pos = get_layout(G, layout)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw graph
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="lightblue",
        node_size=node_size,
        edgecolors="black",
        linewidths=2,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="black",
        width=edge_width,
        arrowsize=20,
        arrowstyle="->",
        ax=ax,
    )

    # Add graph statistics
    n_nodes = len(G.nodes())
    n_edges = len(G.edges())
    stats_text = f"Nodes: {n_nodes}, Edges: {n_edges}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved graph to: {save_path}")

    if show:
        plt.show()

    return fig


def compare_graphs(
    true_adj: np.ndarray,
    pred_adj: np.ndarray,
    node_names: Optional[List[str]] = None,
    layout: str = "hierarchical",
    title: str = "Ground Truth vs Predicted",
    figsize: Tuple[int, int] = (18, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compare ground truth and predicted graphs side-by-side.

    Args:
        true_adj: Ground truth adjacency matrix
        pred_adj: Predicted adjacency matrix
        node_names: Node labels
        layout: Layout algorithm
        title: Overall title
        figsize: Figure size
        save_path: If provided, save figure to this path
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    G_true = adjacency_to_networkx(true_adj, node_names)
    G_pred = adjacency_to_networkx(pred_adj, node_names)

    # Use same layout for both graphs for easy comparison
    pos = get_layout(G_true, layout)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Ground Truth
    nx.draw_networkx_nodes(
        G_true,
        pos,
        node_color="lightgreen",
        node_size=2000,
        edgecolors="black",
        linewidths=2,
        ax=ax1,
    )
    nx.draw_networkx_labels(G_true, pos, font_size=12, font_weight="bold", ax=ax1)
    nx.draw_networkx_edges(
        G_true,
        pos,
        edge_color="green",
        width=2.0,
        arrowsize=20,
        arrowstyle="->",
        ax=ax1,
    )
    ax1.set_title(
        f"Ground Truth ({len(G_true.edges())} edges)", fontsize=14, fontweight="bold"
    )
    ax1.axis("off")

    # Predicted
    nx.draw_networkx_nodes(
        G_pred,
        pos,
        node_color="lightcoral",
        node_size=2000,
        edgecolors="black",
        linewidths=2,
        ax=ax2,
    )
    nx.draw_networkx_labels(G_pred, pos, font_size=12, font_weight="bold", ax=ax2)
    nx.draw_networkx_edges(
        G_pred, pos, edge_color="red", width=2.0, arrowsize=20, arrowstyle="->", ax=ax2
    )
    ax2.set_title(
        f"Predicted ({len(G_pred.edges())} edges)", fontsize=14, fontweight="bold"
    )
    ax2.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison to: {save_path}")

    if show:
        plt.show()

    return fig


def visualize_graph_differences(
    true_adj: np.ndarray,
    pred_adj: np.ndarray,
    node_names: Optional[List[str]] = None,
    layout: str = "hierarchical",
    title: str = "Graph Differences (TP/FP/FN)",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize differences between ground truth and predicted graphs.

    Edge colors:
    - Green: True Positive (correct edge)
    - Red: False Positive (extra edge)
    - Gray dashed: False Negative (missing edge)

    Args:
        true_adj: Ground truth adjacency matrix
        pred_adj: Predicted adjacency matrix
        node_names: Node labels
        layout: Layout algorithm
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    n_nodes = true_adj.shape[0]
    if node_names is None:
        node_names = [f"G{i}" for i in range(n_nodes)]

    # Create graph with all nodes
    G = nx.DiGraph()
    G.add_nodes_from(node_names)

    # Classify edges
    tp_edges = []  # True Positives
    fp_edges = []  # False Positives
    fn_edges = []  # False Negatives

    for i in range(n_nodes):
        for j in range(n_nodes):
            edge = (node_names[i], node_names[j])

            if true_adj[i, j] == 1 and pred_adj[i, j] == 1:
                tp_edges.append(edge)
                G.add_edge(*edge)
            elif true_adj[i, j] == 0 and pred_adj[i, j] == 1:
                fp_edges.append(edge)
                G.add_edge(*edge)
            elif true_adj[i, j] == 1 and pred_adj[i, j] == 0:
                fn_edges.append(edge)
                G.add_edge(*edge)

    pos = get_layout(G, layout)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="lightblue",
        node_size=2000,
        edgecolors="black",
        linewidths=2,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)

    # Draw edges by type
    if tp_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=tp_edges,
            edge_color="green",
            width=3.0,
            arrowsize=20,
            arrowstyle="->",
            ax=ax,
            label=f"True Positive ({len(tp_edges)})",
        )

    if fp_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=fp_edges,
            edge_color="red",
            width=3.0,
            arrowsize=20,
            arrowstyle="->",
            ax=ax,
            label=f"False Positive ({len(fp_edges)})",
        )

    if fn_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=fn_edges,
            edge_color="gray",
            width=2.0,
            arrowsize=20,
            arrowstyle="->",
            style="dashed",
            alpha=0.5,
            ax=ax,
            label=f"False Negative ({len(fn_edges)})",
        )

    # Calculate metrics
    tp = len(tp_edges)
    fp = len(fp_edges)
    fn = len(fn_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # Add metrics text
    metrics_text = (
        f"Precision: {precision:.3f}\n"
        f"Recall: {recall:.3f}\n"
        f"F1 Score: {f1:.3f}\n"
        f"TP: {tp}, FP: {fp}, FN: {fn}"
    )
    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved difference graph to: {save_path}")

    if show:
        plt.show()

    return fig


def visualize_experiment_results(
    dataset_name: str,
    true_adj: np.ndarray,
    pred_adj: np.ndarray,
    node_names: Optional[List[str]] = None,
    method: str = "FedCDH",
    scenario: str = "Horizontal",
    output_dir: str = "./graphs",
    layout: str = "hierarchical",
    show: bool = False,
) -> Dict[str, str]:
    """
    Generate complete visualization suite for an experiment.

    Creates:
    1. Ground truth graph
    2. Predicted graph
    3. Side-by-side comparison
    4. Difference visualization with metrics

    Args:
        dataset_name: Name of dataset (e.g., "sachs", "dream4_net1")
        true_adj: Ground truth adjacency
        pred_adj: Predicted adjacency
        node_names: Node labels
        method: Algorithm name
        scenario: Experiment scenario
        output_dir: Directory to save visualizations
        layout: Layout algorithm
        show: Whether to display plots

    Returns:
        Dictionary mapping visualization type -> file path
    """
    os.makedirs(output_dir, exist_ok=True)

    # File name prefix
    prefix = f"{dataset_name}_{method}_{scenario}"

    saved_files = {}

    # 1. Ground Truth
    true_path = os.path.join(output_dir, f"{prefix}_ground_truth.png")
    visualize_dag(
        true_adj,
        node_names,
        layout=layout,
        title=f"{dataset_name.upper()} - Ground Truth",
        save_path=true_path,
        show=show,
    )
    plt.close()
    saved_files["ground_truth"] = true_path

    # 2. Predicted Graph
    pred_path = os.path.join(output_dir, f"{prefix}_predicted.png")
    visualize_dag(
        pred_adj,
        node_names,
        layout=layout,
        title=f"{dataset_name.upper()} - Predicted ({method})",
        save_path=pred_path,
        show=show,
    )
    plt.close()
    saved_files["predicted"] = pred_path

    # 3. Side-by-side Comparison
    compare_path = os.path.join(output_dir, f"{prefix}_comparison.png")
    compare_graphs(
        true_adj,
        pred_adj,
        node_names,
        layout=layout,
        title=f"{dataset_name.upper()} - {method} ({scenario})",
        save_path=compare_path,
        show=show,
    )
    plt.close()
    saved_files["comparison"] = compare_path

    # 4. Difference Visualization
    diff_path = os.path.join(output_dir, f"{prefix}_differences.png")
    visualize_graph_differences(
        true_adj,
        pred_adj,
        node_names,
        layout=layout,
        title=f"{dataset_name.upper()} - {method} ({scenario}) - Differences",
        save_path=diff_path,
        show=show,
    )
    plt.close()
    saved_files["differences"] = diff_path

    print(f"\n✅ Generated 4 visualizations for {dataset_name} ({method}, {scenario})")
    print(f"   Saved to: {output_dir}/")

    return saved_files


# ------------------------------------------------------------------------------
# Batch Visualization for Multiple Experiments
# ------------------------------------------------------------------------------


def visualize_multiple_methods(
    dataset_name: str,
    true_adj: np.ndarray,
    predictions: Dict[str, np.ndarray],
    node_names: Optional[List[str]] = None,
    layout: str = "hierarchical",
    figsize: Tuple[int, int] = (20, 10),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compare multiple methods on the same dataset.

    Args:
        dataset_name: Dataset name
        true_adj: Ground truth adjacency
        predictions: Dictionary mapping method_name -> predicted_adjacency
        node_names: Node labels
        layout: Layout algorithm
        figsize: Figure size
        save_path: If provided, save figure to this path
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    n_methods = len(predictions)
    n_cols = min(3, n_methods + 1)  # Max 3 columns
    n_rows = (n_methods + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    # Get consistent layout
    G_true = adjacency_to_networkx(true_adj, node_names)
    pos = get_layout(G_true, layout)

    # Plot ground truth
    ax = axes[0]
    nx.draw_networkx_nodes(
        G_true,
        pos,
        node_color="lightgreen",
        node_size=1500,
        edgecolors="black",
        linewidths=2,
        ax=ax,
    )
    nx.draw_networkx_labels(G_true, pos, font_size=10, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G_true, pos, edge_color="green", width=2.0, arrowsize=15, arrowstyle="->", ax=ax
    )
    ax.set_title(
        f"Ground Truth\n({len(G_true.edges())} edges)", fontsize=12, fontweight="bold"
    )
    ax.axis("off")

    # Plot predictions
    for idx, (method_name, pred_adj) in enumerate(predictions.items(), start=1):
        ax = axes[idx]
        G_pred = adjacency_to_networkx(pred_adj, node_names)

        nx.draw_networkx_nodes(
            G_pred,
            pos,
            node_color="lightcoral",
            node_size=1500,
            edgecolors="black",
            linewidths=2,
            ax=ax,
        )
        nx.draw_networkx_labels(G_pred, pos, font_size=10, font_weight="bold", ax=ax)
        nx.draw_networkx_edges(
            G_pred,
            pos,
            edge_color="red",
            width=2.0,
            arrowsize=15,
            arrowstyle="->",
            ax=ax,
        )

        # Calculate F1
        tp = np.sum((true_adj == 1) & (pred_adj == 1))
        fp = np.sum((true_adj == 0) & (pred_adj == 1))
        fn = np.sum((true_adj == 1) & (pred_adj == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        ax.set_title(
            f"{method_name}\n({len(G_pred.edges())} edges, F1={f1:.3f})",
            fontsize=12,
            fontweight="bold",
        )
        ax.axis("off")

    # Hide unused subplots
    for idx in range(n_methods + 1, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"{dataset_name.upper()} - Method Comparison", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved method comparison to: {save_path}")

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    print("Graph Visualization Utilities - Demo\n")

    # Create example DAG
    print("Creating example DAG...")
    adj = np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    names = ["A", "B", "C", "D", "E"]

    # Demo 1: Single graph
    print("\n1. Visualizing single graph...")
    visualize_dag(adj, names, title="Example DAG", show=True)

    # Demo 2: Comparison
    print("\n2. Comparing graphs...")
    pred_adj = adj.copy()
    pred_adj[0, 1] = 0  # Remove one edge (FN)
    pred_adj[1, 2] = 1  # Add one edge (FP)

    compare_graphs(adj, pred_adj, names, show=True)

    # Demo 3: Differences
    print("\n3. Showing differences...")
    visualize_graph_differences(adj, pred_adj, names, show=True)

    print("\n✅ Demo complete!")
