import sys
import os
import torch
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from causallearn.utils.FedPC import LocalSPNWrapper, GlobalFedSPN, FederatedProduct


def analyze_communication_cost():
    print("=" * 60)
    print("FEDERATED COMMUNICATION COST ANALYSIS")
    print("=" * 60)

    # Parameters
    d = 10
    K = 5
    num_sums = 20
    num_leaves = 20
    depth = 2
    reps = 5

    # 1. Estimate Single Client Model Size
    client = LocalSPNWrapper(
        num_features=d,
        device="cpu",
        num_sums=num_sums,
        num_leaves=num_leaves,
        depth=depth,
        num_repetitions=reps,
    )

    client_bytes = client.get_size_bytes()
    client_kb = client_bytes / 1024
    print(f"[Unit] Single Local SPN Size (d={d}): {client_kb:.2f} KB")

    # 2. Horizontal Scenario (One-Shot Mixture)
    # Server collects K models once.
    horizontal_cost_bytes = client_bytes * K
    horizontal_cost_kb = horizontal_cost_bytes / 1024

    print(f"\n[Horizontal FL] K={K} Clients")
    print(f"  - Upload Cost (One-Shot): {horizontal_cost_kb:.2f} KB")
    print(f"  - Download Cost: ~0 KB (Server keeps model)")
    print(f"  - Total: {horizontal_cost_kb:.2f} KB")

    # Comparison with FedAvg (Deep Learning Baseline)
    # Assume 50 rounds of updates
    # FedAvg Cost = Rounds * 2 (Up/Down) * ModelSize
    rounds = 50
    fedavg_cost_kb = (
        rounds * 2 * horizontal_cost_kb
    )  # (Using sum of models as proxy for global model size if arch is same)
    # Actually FedAvg model size = Single Client Model Size (shared arch).
    fedavg_single_model_kb = client_kb
    fedavg_total_kb = rounds * 2 * K * fedavg_single_model_kb

    print(f"  - VS FedAvg (50 Rounds): {fedavg_total_kb:.2f} KB")
    print(f"  - Efficiency Gain: {fedavg_total_kb / horizontal_cost_kb:.1f}x")

    # 3. Vertical Scenario (Latent Variable)
    # Clients train 'num_clusters' models each.
    # Total Models = K * num_clusters
    num_clusters = 5
    vertical_total_models = K * num_clusters
    # Vertical Features per client
    d_local = d // K
    # Safe depth for d=2 is 1
    client_vert = LocalSPNWrapper(
        num_features=d_local,
        device="cpu",
        num_sums=num_sums,
        num_leaves=num_leaves,
        depth=1,
        num_repetitions=reps,
    )
    client_vert_kb = client_vert.get_size_bytes() / 1024

    vertical_cost_kb = client_vert_kb * vertical_total_models

    print(f"\n[Vertical FL] K={K} Clients, H={num_clusters} Latent Clusters")
    print(f"  - Single Sub-Model Size (d={d_local}): {client_vert_kb:.2f} KB")
    print(f"  - Total Models Uploaded: {vertical_total_models}")
    print(f"  - Total Upload Cost: {vertical_cost_kb:.2f} KB")

    # Comparison: Vertical often requires PSI (Private Set Intersection) or encrypted alignment per row.
    # Our method requires ZERO per-sample alignment after training.

    print("=" * 60)


if __name__ == "__main__":
    analyze_communication_cost()
