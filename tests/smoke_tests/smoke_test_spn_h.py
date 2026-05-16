#!/usr/bin/env python
"""
Quick Smoke Test: FedSPN Horizontal (Your Proposed Method)

Tests FedSPN with horizontal partitioning on a tiny synthetic dataset.

Configuration:
- Dataset: Synthetic ER graph (5 nodes, 200 samples)
- Method: FedSPN-H (SPN-based CI tests, horizontal FL)
- Clients: K=2
- Device: CPU only
- Expected runtime: 2-3 minutes
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
from argparse import Namespace

from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH

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
    nx.draw_networkx_nodes(G, pos, node_color="gold", node_size=1000, alpha=0.9)
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

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "shd": shd,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    logging.info("=" * 80)
    logging.info("FEDSPN-H SMOKE TEST - SPN-based Horizontal Federated Learning")
    logging.info("=" * 80)

    d, n, K, seed = 5, 200, 2, 42
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / f"{timestamp}_spn_h_{d}vars_{n}samples_K{K}"
    output_dir.mkdir(exist_ok=True)

    logging.info(f"Configuration: d={d}, n={n}, K={K}, seed={seed}")
    logging.info(f"Device: CPU")
    logging.info(f"Output: {output_dir}")

    # Generate data
    B_true = generate_tiny_synthetic_dag(d=d, sparsity=0.3, seed=seed)
    X = generate_data_from_dag(B_true, n_samples=n, seed=seed)
    plot_graph(B_true, "True DAG (Ground Truth)", output_dir / "true_graph.png")

    # Partition horizontally
    n_trimmed = (n // K) * K
    X = X[:n_trimmed, :]
    samples_per_client = n_trimmed // K
    X_splits = [
        X[k * samples_per_client : (k + 1) * samples_per_client, :] for k in range(K)
    ]

    # Create context index (client assignments)
    c_indx = np.repeat(np.arange(K), samples_per_client).reshape(-1, 1)

    logging.info(
        f"\nHorizontal partitioning: {K} clients with {samples_per_client} samples each"
    )

    # Create FedCDH args
    args = Namespace(
        K=K,
        d=d,
        n=samples_per_client,  # Samples per client
        scenario="horizontal",
        num_local_clusters=2,
        epochs=50,  # Reduced for speed
        alpha=0.05,
        ci_method="spn",
        model_type="nonparametric",
        device="cpu",
        output_dir=str(output_dir),
        verbose=False,
    )

    # Run FedSPN-H
    logging.info("\nRunning FedSPN-H (this may take 2-3 minutes on CPU)...")
    start_time = time.time()

    try:
        fedcdh = FedCDH(args)
        fedcdh.fit(X_splits, c_indx, B_true)
        runtime = time.time() - start_time

        # Extract discovered graph
        G_pred = fedcdh.est_dag if hasattr(fedcdh, "est_dag") else np.zeros((d, d))
        plot_graph(
            G_pred, "Discovered DAG (FedSPN-H)", output_dir / "discovered_graph.png"
        )

        # Evaluate
        metrics = compute_metrics(G_pred, B_true)

        results = {
            "method": "FedSPN-H",
            "timestamp": timestamp,
            "config": {
                "d": d,
                "n": n,
                "K": K,
                "num_local_clusters": 2,
                "epochs": 50,
                "seed": seed,
                "device": "cpu",
                "scenario": "horizontal",
            },
            "metrics": {
                k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                for k, v in metrics.items()
            },
            "runtime_seconds": float(runtime),
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(output_dir / "summary.txt", "w") as f:
            f.write("FEDSPN-H SMOKE TEST SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Method: FedSPN-H (SPN-based horizontal FL)\n")
            f.write(f"Dataset: Synthetic ({d} vars, {n} samples)\n")
            f.write(f"Clients: K={K}\n")
            f.write(f"Runtime: {runtime:.2f}s\n\n")
            f.write(f"F1:        {metrics['f1']:.3f}\n")
            f.write(f"Precision: {metrics['precision']:.3f}\n")
            f.write(f"Recall:    {metrics['recall']:.3f}\n")
            f.write(f"SHD:       {metrics['shd']}\n")
            f.write(f"TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}\n\n")
            f.write("✓ PASSED\n" if metrics["f1"] > 0.3 else "⚠ MARGINAL\n")

        logging.info(
            f"\nResults: F1={metrics['f1']:.3f}, SHD={metrics['shd']}, Runtime={runtime:.2f}s"
        )
        logging.info(f"✓ Results saved to: {output_dir}")
        logging.info("=" * 80)

    except Exception as e:
        runtime = time.time() - start_time
        logging.error(f"\n✗ SMOKE TEST FAILED after {runtime:.2f}s")
        logging.error(f"Error: {e}")
        import traceback

        traceback.print_exc()

        with open(output_dir / "error.txt", "w") as f:
            f.write(f"Error: {e}\n\n")
            f.write(traceback.format_exc())

        sys.exit(1)


if __name__ == "__main__":
    main()
