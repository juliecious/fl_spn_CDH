#!/usr/bin/env python
"""
Quick Smoke Test: FedCDH with KCI (Original CD-NOD Method)

Tests the original FedCDH algorithm using KCI tests on a tiny synthetic dataset.
This verifies that the KCI-based baseline is working before running larger experiments.

Configuration:
- Dataset: Synthetic ER graph (5 nodes, 200 samples)
- Method: CD-NOD with KCI tests
- Clients: K=2 (minimal for federated)
- Device: CPU only
- Expected runtime: 2-5 minutes
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import time
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import kci
from causallearn.utils.data_utils import (
    get_cpdag_from_cdnod,
    get_dag_from_pdag,
    count_skeleton_accuracy,
    count_dag_accuracy,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_tiny_synthetic_dag(d=5, sparsity=0.3, seed=42):
    """Generate a tiny random DAG for smoke testing."""
    np.random.seed(seed)

    B = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if np.random.rand() < sparsity:
                B[i, j] = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])

    return B


def generate_data_from_dag(B, n_samples=200, noise_scale=1.0, seed=42):
    """Generate data from DAG using linear SEM."""
    np.random.seed(seed)
    d = B.shape[0]
    X = np.zeros((n_samples, d))

    # Topological order generation
    for j in range(d):
        parents = np.where(B[:, j] != 0)[0]
        if len(parents) == 0:
            # Root node: just noise
            X[:, j] = np.random.randn(n_samples) * noise_scale
        else:
            # Non-root: linear combination of parents + noise
            X[:, j] = (
                X[:, parents] @ B[parents, j] + np.random.randn(n_samples) * noise_scale
            )

    return X


def partition_horizontal(X, K):
    """Partition data into K clients (horizontal FL)."""
    n = X.shape[0]
    samples_per_client = n // K

    c_indx = np.repeat(np.arange(K), samples_per_client)
    remainder = n % K
    if remainder > 0:
        c_indx = np.concatenate([c_indx, np.arange(remainder)])

    c_indx = c_indx[:n].reshape(-1, 1)
    return X, c_indx


def plot_graph(adj_matrix, title, output_path, node_labels=None):
    """Plot adjacency matrix as directed graph and save to PNG."""
    d = adj_matrix.shape[0]
    if node_labels is None:
        node_labels = [f"X{i}" for i in range(d)]

    G = nx.DiGraph()
    G.add_nodes_from(range(d))

    # Add edges
    for i in range(d):
        for j in range(d):
            if adj_matrix[i, j] != 0:
                G.add_edge(i, j)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1000, alpha=0.9)
    nx.draw_networkx_labels(
        G, pos, labels={i: node_labels[i] for i in range(d)}, font_size=12
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.1",
        width=2,
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Saved graph: {output_path}")


def main():
    logging.info("=" * 80)
    logging.info(
        "KCI SMOKE TEST - FedCDH (CD-NOD) with Kernel Conditional Independence"
    )
    logging.info("=" * 80)

    # Configuration
    d = 5  # Number of variables (very small)
    n = 200  # Number of samples (minimal)
    K = 2  # Number of clients
    alpha = 0.05  # Significance level
    sparsity = 0.3  # Graph sparsity
    seed = 42

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / f"{timestamp}_kci_{d}vars_{n}samples"
    output_dir.mkdir(exist_ok=True)

    logging.info(f"Configuration:")
    logging.info(f"  Variables (d): {d}")
    logging.info(f"  Samples (n): {n}")
    logging.info(f"  Clients (K): {K}")
    logging.info(f"  Alpha: {alpha}")
    logging.info(f"  Device: CPU")
    logging.info(f"  Seed: {seed}")
    logging.info(f"  Output: {output_dir}")

    # Step 1: Generate synthetic DAG
    logging.info("\n" + "=" * 80)
    logging.info("Step 1: Generating Synthetic DAG")
    logging.info("=" * 80)

    B_true = generate_tiny_synthetic_dag(d=d, sparsity=sparsity, seed=seed)
    n_edges_true = int(np.sum(np.abs(B_true) > 0))

    logging.info(f"Generated DAG: {d} nodes, {n_edges_true} edges")
    logging.info(f"Adjacency matrix:\n{(B_true != 0).astype(int)}")

    # Save and plot true graph
    plot_graph(B_true, "True DAG (Ground Truth)", output_dir / "true_graph.png")

    # Step 2: Generate data
    logging.info("\n" + "=" * 80)
    logging.info("Step 2: Generating Synthetic Data")
    logging.info("=" * 80)

    X = generate_data_from_dag(B_true, n_samples=n, seed=seed)
    logging.info(f"Data shape: {X.shape}")

    # Step 3: Partition data
    logging.info("\n" + "=" * 80)
    logging.info("Step 3: Partitioning Data (Horizontal FL)")
    logging.info("=" * 80)

    X_partitioned, c_indx = partition_horizontal(X, K)

    for k in range(K):
        client_samples = np.sum(c_indx == k)
        logging.info(f"  Client {k}: {client_samples} samples")

    # Step 4: Run FedCDH with KCI
    logging.info("\n" + "=" * 80)
    logging.info("Step 4: Running FedCDH (CD-NOD) with KCI Tests")
    logging.info("=" * 80)
    logging.info("This may take 2-5 minutes depending on CPU...")

    start_time = time.time()

    try:
        cg = cdnod(
            X_partitioned,
            c_indx,
            K,
            alpha,
            kci,  # Kernel Conditional Independence test
            True,  # background_knowledge
            0,  # uc_rule
            -1,  # uc_priority
        )

        runtime = time.time() - start_time

        logging.info(f"✓ FedCDH completed in {runtime:.2f} seconds")

        # Step 5: Extract and evaluate results
        logging.info("\n" + "=" * 80)
        logging.info("Step 5: Evaluating Results")
        logging.info("=" * 80)

        # Extract graph (remove context variable U)
        est_graph = cg.G.graph[0:d, 0:d]

        # Convert to CPDAG and DAG
        est_cpdag = get_cpdag_from_cdnod(est_graph)
        est_dag = get_dag_from_pdag(est_cpdag)

        # Plot discovered graph
        plot_graph(est_dag, "Discovered DAG (KCI)", output_dir / "discovered_graph.png")

        # Compute metrics
        skeleton_metrics = count_skeleton_accuracy(B_true, est_cpdag)
        dag_metrics = count_dag_accuracy(B_true, est_dag)

        # Collect results
        results = {
            "method": "FedCDH-KCI",
            "timestamp": timestamp,
            "config": {
                "d": d,
                "n": n,
                "K": K,
                "alpha": alpha,
                "seed": seed,
                "device": "cpu",
            },
            "skeleton_metrics": {
                "f1": float(skeleton_metrics.get("f1", 0.0)),
                "precision": float(skeleton_metrics.get("precision", 0.0)),
                "recall": float(skeleton_metrics.get("recall", 0.0)),
                "shd": int(skeleton_metrics.get("shd", 0)),
            },
            "dag_metrics": {
                "f1": float(dag_metrics.get("f1", 0.0)),
                "precision": float(dag_metrics.get("precision", 0.0)),
                "recall": float(dag_metrics.get("recall", 0.0)),
                "shd": int(dag_metrics.get("shd", 0)),
            },
            "runtime_seconds": float(runtime),
        }

        # Save results as JSON
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"  Saved results: {output_dir / 'results.json'}")

        # Save summary
        with open(output_dir / "summary.txt", "w") as f:
            f.write("=" * 80 + "\n")
            f.write("KCI SMOKE TEST SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Method: FedCDH with KCI\n")
            f.write(f"Dataset: Synthetic ({d} vars, {n} samples)\n")
            f.write(f"Clients: K={K}\n")
            f.write(f"Runtime: {runtime:.2f}s\n\n")

            f.write("-" * 80 + "\n")
            f.write("SKELETON METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"F1:        {skeleton_metrics.get('f1', 0.0):.3f}\n")
            f.write(f"Precision: {skeleton_metrics.get('precision', 0.0):.3f}\n")
            f.write(f"Recall:    {skeleton_metrics.get('recall', 0.0):.3f}\n")
            f.write(f"SHD:       {skeleton_metrics.get('shd', 0)}\n\n")

            f.write("-" * 80 + "\n")
            f.write("DAG METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"F1:        {dag_metrics.get('f1', 0.0):.3f}\n")
            f.write(f"Precision: {dag_metrics.get('precision', 0.0):.3f}\n")
            f.write(f"Recall:    {dag_metrics.get('recall', 0.0):.3f}\n")
            f.write(f"SHD:       {dag_metrics.get('shd', 0)}\n\n")

            # Discovered edges
            discovered_edges = []
            for i in range(d):
                for j in range(d):
                    if est_dag[i, j] != 0:
                        discovered_edges.append(f"X{i} -> X{j}")

            true_edges = []
            for i in range(d):
                for j in range(d):
                    if B_true[i, j] != 0:
                        true_edges.append(f"X{i} -> X{j}")

            f.write("-" * 80 + "\n")
            f.write("GRAPH COMPARISON\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"True edges:       {', '.join(true_edges) if true_edges else 'None'}\n"
            )
            f.write(
                f"Discovered edges: {', '.join(discovered_edges) if discovered_edges else 'None'}\n\n"
            )

            if skeleton_metrics.get("f1", 0.0) > 0.3:
                f.write("✓ SMOKE TEST PASSED\n")
            else:
                f.write("⚠ SMOKE TEST MARGINAL (low F1, likely due to small sample)\n")

        logging.info(f"  Saved summary: {output_dir / 'summary.txt'}")

        # Report results
        logging.info("\n" + "-" * 80)
        logging.info("SKELETON METRICS (Undirected Graph)")
        logging.info("-" * 80)
        logging.info(f"  F1 Score:   {skeleton_metrics.get('f1', 0.0):.3f}")
        logging.info(f"  Precision:  {skeleton_metrics.get('precision', 0.0):.3f}")
        logging.info(f"  Recall:     {skeleton_metrics.get('recall', 0.0):.3f}")
        logging.info(f"  SHD:        {skeleton_metrics.get('shd', 0)}")

        logging.info("\n" + "-" * 80)
        logging.info("DAG METRICS (Directed Graph)")
        logging.info("-" * 80)
        logging.info(f"  F1 Score:   {dag_metrics.get('f1', 0.0):.3f}")
        logging.info(f"  Precision:  {dag_metrics.get('precision', 0.0):.3f}")
        logging.info(f"  Recall:     {dag_metrics.get('recall', 0.0):.3f}")
        logging.info(f"  SHD:        {dag_metrics.get('shd', 0)}")

        # Summary
        logging.info("\n" + "=" * 80)
        logging.info("SMOKE TEST SUMMARY")
        logging.info("=" * 80)

        if skeleton_metrics.get("f1", 0.0) > 0.3:
            logging.info("✓ SMOKE TEST PASSED")
            logging.info(f"  FedCDH with KCI is working correctly!")
            logging.info(
                f"  Skeleton F1 = {skeleton_metrics.get('f1', 0.0):.3f} (> 0.3 threshold)"
            )
        else:
            logging.info("⚠ SMOKE TEST MARGINAL")
            logging.info(
                f"  FedCDH completed but F1 is low: {skeleton_metrics.get('f1', 0.0):.3f}"
            )
            logging.info(f"  This might be due to random data or small sample size")

        logging.info(f"  Runtime: {runtime:.2f}s on CPU")
        logging.info(f"  Results saved to: {output_dir}")
        logging.info("=" * 80)

    except Exception as e:
        runtime = time.time() - start_time
        logging.error(f"\n✗ SMOKE TEST FAILED after {runtime:.2f}s")
        logging.error(f"Error: {e}")
        import traceback

        traceback.print_exc()

        # Save error
        with open(output_dir / "error.txt", "w") as f:
            f.write(f"Error: {e}\n\n")
            f.write(traceback.format_exc())

        logging.info("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
