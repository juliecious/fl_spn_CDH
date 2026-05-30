#!/usr/bin/env python
"""
Asia Horizontal FL Experiment - All Bug Fixes Applied

Tests the complete pipeline on Asia dataset:
1. Load Asia data (8 variables, ~10K samples)
2. Split horizontally across 3 clients
3. Train with structural synchronization (Bug 1 & 2 fixes)
4. DAG search with CI testing (Bug 3 fix)
5. Evaluate SHD

Expected: SHD < 5, F1 > 0.70
"""

import sys
import os
import logging
import time
import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add project to path
sys.path.insert(0, "/Users/M279402/PycharmProjects/fl_spn_CDH")


def load_or_generate_asia():
    """Load Asia dataset or generate synthetic version."""
    data_path = "data/benchmarks/asia_data.npy"
    dag_path = "data/benchmarks/asia_dag.npy"

    if os.path.exists(data_path) and os.path.exists(dag_path):
        logging.info(f"Loading Asia dataset from {data_path}")
        data = np.load(data_path)
        true_dag = np.load(dag_path)
        return data, true_dag

    # Generate synthetic Asia-like data
    logging.warning("Asia dataset not found, generating synthetic version")
    np.random.seed(42)
    n = 5000

    # Asia DAG structure (simplified):
    # 0:A → 1:T → 3:E ← 2:L
    #              ↓
    #             4:X
    # Plus independent: 5:S → 6:B → 7:D

    A = np.random.binomial(1, 0.01, n).astype(float)
    T = (A * 0.05 + (1 - A) * 0.01) * np.random.binomial(1, 0.5, n)
    L = np.random.binomial(1, 0.055, n).astype(float)
    E = np.maximum(T, L).astype(float)
    X = E * 0.98 + np.random.randn(n) * 0.1
    S = np.random.binomial(1, 0.5, n).astype(float)
    B = (S * 0.6 + (1 - S) * 0.3) * np.random.binomial(1, 0.5, n)
    D = (E + B > 0.5).astype(float) + np.random.randn(n) * 0.1

    data = np.column_stack([A, T, L, E, X, S, B, D])

    # True DAG (undirected skeleton)
    true_dag = np.array(
        [
            [0, 1, 0, 0, 0, 0, 0, 0],  # A → T
            [1, 0, 0, 1, 0, 0, 0, 0],  # T → E
            [0, 0, 0, 1, 0, 0, 0, 0],  # L → E
            [0, 1, 1, 0, 1, 0, 0, 1],  # E → X, E → D
            [0, 0, 0, 1, 0, 0, 0, 0],  # X
            [0, 0, 0, 0, 0, 0, 1, 0],  # S → B
            [0, 0, 0, 0, 0, 1, 0, 1],  # B → D
            [0, 0, 0, 1, 0, 0, 1, 0],  # D
        ]
    )

    return data, true_dag


def main():
    start_time = time.time()

    logging.info("=" * 60)
    logging.info("Asia Horizontal FL Experiment (Bug Fixes Applied)")
    logging.info("=" * 60)

    # Step 1: Load data
    logging.info("\n[Step 1/7] Loading Asia dataset...")
    data, true_dag = load_or_generate_asia()
    logging.info(f"  Data shape: {data.shape}")
    logging.info(f"  True edges: {true_dag.sum() // 2}")

    # Step 2: Import modules
    logging.info("\n[Step 2/7] Importing modules...")
    from causallearn.utils.spn.federated import (
        broadcast_structure_to_clients,
        GlobalFedSPN,
    )
    from causallearn.utils.spn.federated.client_init import (
        initialize_heterogeneous_clients,
        train_clients_locally,
        validate_structural_alignment_detailed,
    )
    from causallearn.utils.spn.causal.ci_testing import (
        greedy_dag_search_via_circuit_ci,
    )

    # Step 3: Split data (horizontal)
    logging.info("\n[Step 3/7] Splitting data horizontally (3 clients)...")
    n = len(data)
    client_data = [data[: n // 3], data[n // 3 : 2 * n // 3], data[2 * n // 3 :]]
    logging.info(f"  Client sizes: {[len(d) for d in client_data]}")

    # Step 4: Broadcast template (Bug 1 fix)
    logging.info("\n[Step 4/7] Broadcasting structure template...")
    template, client_seeds = broadcast_structure_to_clients(
        num_features=8,
        num_clients=3,
        global_seed=42,
        depth=3,
        num_sums=20,
        num_leaves=20,
    )
    logging.info(f"  Structure type: {template['config']['structure']}")
    logging.info(f"  Client seeds: {client_seeds}")

    # Step 5: Initialize & train clients (Bug 2 fix)
    logging.info("\n[Step 5/7] Initializing clients with heterogeneous parameters...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"  Device: {device}")

    clients = initialize_heterogeneous_clients(
        client_data,
        template,
        client_seeds,
        device=device,
        verbose=True,
    )

    # Validate alignment
    is_valid = validate_structural_alignment_detailed(clients, verbose=True)
    if not is_valid:
        logging.error("  ❌ Clients not properly aligned!")
        return

    logging.info("\n  Training clients locally (50 epochs)...")
    final_lls = train_clients_locally(
        clients,
        client_data,
        epochs=50,
        lr=0.005,
        verbose=True,
    )
    logging.info(f"  Final LLs: {[f'{ll:.2f}' for ll in final_lls]}")

    # Step 6: Build global mixture
    logging.info("\n[Step 6/7] Building global federated mixture...")
    sample_sizes = [len(d) for d in client_data]
    weights = [n / sum(sample_sizes) for n in sample_sizes]

    global_spn = GlobalFedSPN(
        components=clients,
        weights=weights,
        strategy="mixture",
        device=device,
    )
    logging.info(f"  Mixture weights: {[f'{w:.3f}' for w in weights]}")

    # Step 7: DAG search with CI testing (Bug 3 fix)
    logging.info("\n[Step 7/7] Running CI-based DAG search...")
    logging.info("  Testing independence with threshold=0.03...")

    learned_skeleton = greedy_dag_search_via_circuit_ci(
        global_spn,
        validation_data=data,
        threshold=0.03,
        max_conditioning_size=3,
        verbose=True,
    )

    logging.info(f"\n  Learned skeleton:\n{learned_skeleton}")
    logging.info(f"  Number of edges: {learned_skeleton.sum() // 2}")

    # Evaluation
    logging.info("\n" + "=" * 60)
    logging.info("RESULTS")
    logging.info("=" * 60)

    # SHD
    shd = np.abs(true_dag - learned_skeleton).sum() // 2
    logging.info(f"Structural Hamming Distance (SHD): {shd}")

    # Precision/Recall/F1
    true_edges = true_dag > 0
    learned_edges = learned_skeleton > 0

    tp = (true_edges & learned_edges).sum() // 2
    fp = (learned_edges & ~true_edges).sum() // 2
    fn = (true_edges & ~learned_edges).sum() // 2

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    logging.info(f"True Positive edges: {tp}")
    logging.info(f"False Positive edges: {fp}")
    logging.info(f"False Negative edges: {fn}")
    logging.info(f"Precision: {precision:.3f}")
    logging.info(f"Recall: {recall:.3f}")
    logging.info(f"F1 Score: {f1:.3f}")

    # Success criteria
    logging.info("\n" + "=" * 60)
    logging.info("SUCCESS CRITERIA")
    logging.info("=" * 60)

    target_shd = 5
    target_f1 = 0.70

    shd_pass = "✅ PASS" if shd < target_shd else "❌ FAIL"
    f1_pass = "✅ PASS" if f1 > target_f1 else "❌ FAIL"

    logging.info(f"SHD < {target_shd}: {shd_pass} (got {shd})")
    logging.info(f"F1 > {target_f1}: {f1_pass} (got {f1:.3f})")

    elapsed = time.time() - start_time
    logging.info(f"\nTotal time: {elapsed:.1f} seconds")

    if shd < target_shd and f1 > target_f1:
        logging.info("\n🎉 EXPERIMENT SUCCESS! All criteria met.")
        return 0
    else:
        logging.warning("\n⚠️  EXPERIMENT INCOMPLETE: Some criteria not met.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
