#!/usr/bin/env python
"""
Quick Smoke Test: FCI (Fast Causal Inference)

Tests the FCI algorithm (centralized baseline) on a tiny synthetic dataset.

Configuration:
- Dataset: Synthetic ER graph (5 nodes, 200 samples)
- Method: FCI (constraint-based, handles latent confounders)
- Device: CPU only
- Expected runtime: < 1 minute
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import time
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from causallearn.search.ConstraintBased.FCI import fci

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_tiny_synthetic_dag(d=5, sparsity=0.3, seed=42):
    np.random.seed(seed)
    B = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if np.random.rand() < sparsity:
                B[i, j] = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
    return B


def generate_data_from_dag(B, n_samples=200, noise_scale=1.0, seed=42):
    np.random.seed(seed)
    d = B.shape[0]
    X = np.zeros((n_samples, d))
    for j in range(d):
        parents = np.where(B[:, j] != 0)[0]
        if len(parents) == 0:
            X[:, j] = np.random.randn(n_samples) * noise_scale
        else:
            X[:, j] = (
                X[:, parents] @ B[parents, j] + np.random.randn(n_samples) * noise_scale
            )
    return X


def plot_graph(adj_matrix, title, output_path, node_labels=None):
    d = adj_matrix.shape[0]
    if node_labels is None:
        node_labels = [f"X{i}" for i in range(d)]

    G = nx.DiGraph()
    G.add_nodes_from(range(d))

    for i in range(d):
        for j in range(d):
            if adj_matrix[i, j] != 0:
                G.add_edge(i, j)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color="lightcoral", node_size=1000, alpha=0.9)
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


def compute_metrics(G_pred, B_true):
    pred_skeleton = (np.abs(G_pred) + np.abs(G_pred.T)) > 0
    true_skeleton = (np.abs(B_true) + np.abs(B_true.T)) > 0
    np.fill_diagonal(pred_skeleton, 0)
    np.fill_diagonal(true_skeleton, 0)

    pred_edges = pred_skeleton[np.triu_indices_from(pred_skeleton, k=1)]
    true_edges = true_skeleton[np.triu_indices_from(true_skeleton, k=1)]

    tp = np.sum((pred_edges == 1) & (true_edges == 1))
    fp = np.sum((pred_edges == 1) & (true_edges == 0))
    fn = np.sum((pred_edges == 0) & (true_edges == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    shd = fp + fn

    return {"f1": f1, "precision": precision, "recall": recall, "shd": shd}


def main():
    logging.info("=" * 80)
    logging.info("FCI SMOKE TEST - Fast Causal Inference (Centralized)")
    logging.info("=" * 80)

    d, n, alpha, seed = 5, 200, 0.05, 42
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / f"{timestamp}_fci_{d}vars_{n}samples"
    output_dir.mkdir(exist_ok=True)

    logging.info(f"Configuration: d={d}, n={n}, alpha={alpha}, seed={seed}")
    logging.info(f"Output: {output_dir}")

    # Generate data
    B_true = generate_tiny_synthetic_dag(d=d, sparsity=0.3, seed=seed)
    X = generate_data_from_dag(B_true, n_samples=n, seed=seed)
    plot_graph(B_true, "True DAG (Ground Truth)", output_dir / "true_graph.png")

    # Run FCI
    logging.info("\nRunning FCI...")
    start_time = time.time()
    G, edges = fci(X, "fisherz", alpha=alpha, verbose=False, show_progress=False)
    runtime = time.time() - start_time
    G_pred = G.graph

    plot_graph(G_pred, "Discovered PAG (FCI)", output_dir / "discovered_graph.png")

    # Evaluate
    metrics = compute_metrics(G_pred, B_true)

    results = {
        "method": "FCI",
        "timestamp": timestamp,
        "config": {"d": d, "n": n, "alpha": alpha, "seed": seed, "device": "cpu"},
        "metrics": {
            k: float(v) if isinstance(v, (np.floating, float)) else int(v)
            for k, v in metrics.items()
        },
        "runtime_seconds": float(runtime),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "summary.txt", "w") as f:
        f.write("FCI SMOKE TEST SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"F1:        {metrics['f1']:.3f}\n")
        f.write(f"Precision: {metrics['precision']:.3f}\n")
        f.write(f"Recall:    {metrics['recall']:.3f}\n")
        f.write(f"SHD:       {metrics['shd']}\n")
        f.write(f"Runtime:   {runtime:.2f}s\n\n")
        f.write("✓ PASSED\n" if metrics["f1"] > 0.3 else "⚠ MARGINAL\n")

    logging.info(
        f"\nResults: F1={metrics['f1']:.3f}, SHD={metrics['shd']}, Runtime={runtime:.2f}s"
    )
    logging.info(f"✓ Results saved to: {output_dir}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
