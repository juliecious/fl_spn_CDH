"""
FedCDH Causal Graph Visualization Utilities.

Provides functions to visualize discovered causal DAGs after FedCDH completes
skeleton discovery and direction recovery.
"""

import numpy as np
import os
import sys
from typing import Optional, List

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.utils.graph_visualization import (
    visualize_dag,
    compare_graphs,
    visualize_graph_differences,
)


def visualize_fedcdh_result(
    true_dag: np.ndarray,
    est_dag: np.ndarray,
    node_names: Optional[List[str]] = None,
    dataset_name: str = "experiment",
    output_dir: str = "./graphs",
    show: bool = False,
) -> str:
    """
    Visualize the final discovered causal DAG from FedCDH.

    Creates visualizations showing:
    1. Ground truth DAG
    2. Estimated DAG
    3. Side-by-side comparison
    4. Difference plot with TP/FP/FN edges

    Args:
        true_dag: Ground truth adjacency matrix (d x d)
        est_dag: Estimated DAG from FedCDH (d x d)
        node_names: List of variable names (default: G0, G1, ...)
        dataset_name: Name for output files
        output_dir: Directory to save visualizations
        show: Whether to display plots interactively

    Returns:
        Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    d = true_dag.shape[0]
    if node_names is None:
        node_names = [f"G{i}" for i in range(d)]

    prefix = os.path.join(output_dir, dataset_name)

    print(f"\n{'='*60}")
    print(f"Visualizing FedCDH Results: {dataset_name.upper()}")
    print(f"{'='*60}")

    # 1. Ground truth
    visualize_dag(
        true_dag,
        node_names,
        layout="hierarchical",
        title=f"{dataset_name.upper()} - Ground Truth DAG",
        save_path=f"{prefix}_true_dag.png",
        show=show,
    )
    print(f"✓ Saved ground truth: {prefix}_true_dag.png")

    # 2. Estimated DAG
    visualize_dag(
        est_dag,
        node_names,
        layout="hierarchical",
        title=f"{dataset_name.upper()} - Discovered DAG (FedCDH)",
        save_path=f"{prefix}_estimated_dag.png",
        show=show,
    )
    print(f"✓ Saved estimated DAG: {prefix}_estimated_dag.png")

    # 3. Side-by-side comparison
    compare_graphs(
        true_dag,
        est_dag,
        node_names,
        layout="hierarchical",
        title=f"{dataset_name.upper()} - Causal Discovery Comparison",
        save_path=f"{prefix}_comparison.png",
        show=show,
    )
    print(f"✓ Saved comparison: {prefix}_comparison.png")

    # 4. Difference plot
    visualize_graph_differences(
        true_dag,
        est_dag,
        node_names,
        layout="hierarchical",
        title=f"{dataset_name.upper()} - Discovery Errors (TP/FP/FN)",
        save_path=f"{prefix}_differences.png",
        show=show,
    )
    print(f"✓ Saved differences: {prefix}_differences.png")

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}/")
    print(f"{'='*60}\n")

    return output_dir


def batch_visualize_results(
    results: List[dict],
    dataset_names: List[str],
    node_names_list: List[List[str]],
    output_dir: str = "./batch_graphs",
    show: bool = False,
) -> None:
    """
    Batch visualize multiple FedCDH experiment results.

    Args:
        results: List of result dicts from FedCDH.learn() with return_graphs=True
        dataset_names: List of dataset names
        node_names_list: List of node name lists for each dataset
        output_dir: Directory to save all visualizations
        show: Whether to display plots
    """
    print(f"\n{'='*60}")
    print(f"BATCH VISUALIZATION: {len(results)} experiments")
    print(f"{'='*60}\n")

    for i, (result, dataset_name, node_names) in enumerate(
        zip(results, dataset_names, node_names_list), 1
    ):
        print(f"[{i}/{len(results)}] Processing {dataset_name}...")

        if "est_dag" not in result or "true_dag" not in result:
            print(f"  ⚠ Skipping {dataset_name}: Missing graph data")
            print(f"     Set args.return_graphs=True in FedCDH.learn()")
            continue

        visualize_fedcdh_result(
            true_dag=result["true_dag"],
            est_dag=result["est_dag"],
            node_names=node_names,
            dataset_name=dataset_name,
            output_dir=output_dir,
            show=show,
        )

    print(f"\n{'='*60}")
    print(f"✅ Batch visualization complete!")
    print(f"   Total: {len(results)} experiments")
    print(f"   Output: {output_dir}/")
    print(f"{'='*60}\n")


def visualize_discovered_graph(
    est_dag: np.ndarray,
    node_names: Optional[List[str]] = None,
    dataset_name: str = "discovered",
    output_dir: str = "./graphs",
    show: bool = True,
    title: Optional[str] = None,
) -> str:
    """
    Visualize ONLY the discovered causal DAG (real-world scenario without ground truth).

    Use this when you don't have ground truth and just want to see what FedCDH discovered.
    Perfect for real-world applications where the true causal structure is unknown.

    Args:
        est_dag: Discovered DAG adjacency matrix (d x d)
        node_names: List of variable names (default: G0, G1, ...)
        dataset_name: Name for output file
        output_dir: Directory to save visualization
        show: Whether to display plot interactively
        title: Custom plot title (default: auto-generated)

    Returns:
        Path to saved figure
    """
    os.makedirs(output_dir, exist_ok=True)

    d = est_dag.shape[0]
    if node_names is None:
        node_names = [f"G{i}" for i in range(d)]

    if title is None:
        title = f"{dataset_name.upper()} - Discovered Causal Structure"

    output_path = os.path.join(output_dir, f"{dataset_name}_discovered_dag.png")

    print(f"\n{'='*60}")
    print(f"Visualizing Discovered Causal Graph: {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"  Variables: {d}")
    print(f"  Edges discovered: {int(est_dag.sum())}")

    # Count node statistics
    in_degrees = est_dag.sum(axis=0)
    out_degrees = est_dag.sum(axis=1)

    print(f"\n  Root nodes (no parents): {list(np.where(in_degrees == 0)[0])}")
    print(f"  Leaf nodes (no children): {list(np.where(out_degrees == 0)[0])}")
    print(f"  Max in-degree: {int(in_degrees.max())}")
    print(f"  Max out-degree: {int(out_degrees.max())}")

    # Visualize
    visualize_dag(
        est_dag,
        node_names,
        layout="hierarchical",
        title=title,
        save_path=output_path,
        show=show,
        figsize=(12, 10),
    )

    print(f"\n{'='*60}")
    print(f"✓ Visualization saved: {output_path}")
    print(f"{'='*60}\n")

    return output_path


def visualize_with_cpdag(
    est_cpdag: np.ndarray,
    est_dag: np.ndarray,
    node_names: Optional[List[str]] = None,
    dataset_name: str = "analysis",
    output_dir: str = "./graphs",
    show: bool = False,
) -> str:
    """
    Visualize both CPDAG (partially oriented) and final DAG.

    The CPDAG shows which edges are uncertain (undirected) vs certain (directed).
    Useful for understanding the confidence in causal directions.

    Args:
        est_cpdag: CPDAG matrix (-1/-1 = undirected, 1/-1 = directed)
        est_dag: Final DAG after orientation
        node_names: Variable names
        dataset_name: Name for output
        output_dir: Save directory
        show: Display plots

    Returns:
        Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    d = est_cpdag.shape[0]
    if node_names is None:
        node_names = [f"G{i}" for i in range(d)]

    print(f"\n{'='*60}")
    print(f"Causal Orientation Analysis: {dataset_name.upper()}")
    print(f"{'='*60}")

    # Count edge types in CPDAG
    directed_edges = 0
    undirected_edges = 0

    for i in range(d):
        for j in range(i + 1, d):
            if est_cpdag[i, j] == 1 and est_cpdag[j, i] == -1:
                directed_edges += 1
            elif est_cpdag[j, i] == 1 and est_cpdag[i, j] == -1:
                directed_edges += 1
            elif est_cpdag[i, j] == -1 and est_cpdag[j, i] == -1:
                undirected_edges += 1

    print(f"\n  Directed edges in CPDAG: {directed_edges}")
    print(f"  Undirected edges in CPDAG: {undirected_edges}")
    print(f"  Total edges: {directed_edges + undirected_edges}")
    print(
        f"  Orientation certainty: {directed_edges / (directed_edges + undirected_edges) * 100:.1f}%"
        if (directed_edges + undirected_edges) > 0
        else "  N/A"
    )

    # Visualize CPDAG
    cpdag_skeleton = (np.abs(est_cpdag) + np.abs(est_cpdag.T)) > 0
    visualize_dag(
        cpdag_skeleton.astype(int),
        node_names,
        layout="hierarchical",
        title=f"{dataset_name.upper()} - CPDAG (Partially Oriented Graph)",
        save_path=f"{output_dir}/{dataset_name}_cpdag.png",
        show=show,
    )
    print(f"✓ Saved CPDAG: {output_dir}/{dataset_name}_cpdag.png")

    # Visualize final DAG
    visualize_dag(
        est_dag,
        node_names,
        layout="hierarchical",
        title=f"{dataset_name.upper()} - Final DAG (Fully Oriented)",
        save_path=f"{output_dir}/{dataset_name}_final_dag.png",
        show=show,
    )
    print(f"✓ Saved final DAG: {output_dir}/{dataset_name}_final_dag.png")

    print(f"\n{'='*60}")
    print(f"Analysis complete! Saved to: {output_dir}/")
    print(f"{'='*60}\n")

    return output_dir


if __name__ == "__main__":
    """Demo: Visualize with and without ground truth."""
    print("=" * 60)
    print("Demo 1: Benchmarking Mode (with ground truth)")
    print("=" * 60)

    # Create simple test DAG
    true_dag = np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])

    # Simulate discovered DAG with some errors
    est_dag = true_dag.copy()
    est_dag[0, 1] = 0  # False negative (missed edge)
    est_dag[1, 2] = 1  # False positive (extra edge)

    node_names = ["A", "B", "C", "D"]

    # Visualize with ground truth (benchmarking)
    visualize_fedcdh_result(
        true_dag=true_dag,
        est_dag=est_dag,
        node_names=node_names,
        dataset_name="benchmark_demo",
        output_dir="/tmp/fedcdh_demo",
        show=False,
    )

    print("\n" + "=" * 60)
    print("Demo 2: Real-World Mode (no ground truth)")
    print("=" * 60)

    # Visualize only discovered graph (real-world scenario)
    visualize_discovered_graph(
        est_dag=est_dag,
        node_names=node_names,
        dataset_name="real_world_demo",
        output_dir="/tmp/fedcdh_demo",
        show=False,
    )

    print("\n✅ Demos complete! Check /tmp/fedcdh_demo/")
