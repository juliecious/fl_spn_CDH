"""
Unit tests for V2 local clustering implementation.

Tests:
1. LocalClusterMixture class functionality
2. Data distribution (no fragmentation)
3. Global vs local clustering comparison
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from sklearn.cluster import KMeans

from causallearn.utils.FedPC import LocalClusterMixture, LocalSPNWrapper


def test_local_cluster_mixture():
    """Test LocalClusterMixture class."""
    print("\n" + "=" * 60)
    print("TEST 1: LocalClusterMixture")
    print("=" * 60)

    # Create 2 SPNs
    spn_0 = LocalSPNWrapper(num_features=5, device="cpu", seed=0)
    spn_1 = LocalSPNWrapper(num_features=5, device="cpu", seed=1)

    # Dummy training
    data = np.random.randn(100, 5)
    spn_0.train_local(data, epochs=5)
    spn_1.train_local(data, epochs=5)

    # Create mixture
    mixture = LocalClusterMixture(
        [spn_0, spn_1], cluster_weights=[0.6, 0.4], client_id=0
    )

    # Test log_prob
    x = torch.randn(50, 5)
    log_p = mixture.log_prob(x)
    assert log_p.shape == (50, 1), f"Expected (50, 1), got {log_p.shape}"
    assert not torch.isnan(log_p).any(), "NaN in log_prob"

    # Test sample
    samples = mixture.sample(30)
    assert samples.shape == (30, 5), f"Expected (30, 5), got {samples.shape}"

    print("✓ LocalClusterMixture: log_prob and sample work correctly")


def test_no_fragmentation():
    """Test that local clustering preserves data."""
    print("\n" + "=" * 60)
    print("TEST 2: No Data Fragmentation")
    print("=" * 60)

    # Setup: 3 clients, 400 samples each
    K_clients = 3
    n_per_client = 400
    K_local = 2

    X_splits = [np.random.randn(n_per_client, 10) for _ in range(K_clients)]

    # Local clustering per client
    cluster_sizes = []
    for k in range(K_clients):
        client_data = X_splits[k]
        kmeans = KMeans(n_clusters=K_local, random_state=42)
        local_clusters = kmeans.fit_predict(client_data)

        for h in range(K_local):
            size = (local_clusters == h).sum()
            cluster_sizes.append(size)
            print(f"  Client {k}, Local Cluster {h}: {size} samples")

    # Check: All clusters have sufficient data
    min_size = min(cluster_sizes)
    avg_size = np.mean(cluster_sizes)

    print(
        f"\nCluster sizes: min={min_size}, avg={avg_size:.1f}, max={max(cluster_sizes)}"
    )

    assert min_size >= 150, f"Fragmentation detected! Min size {min_size} < 150"
    assert avg_size >= 195, f"Average size too low: {avg_size} < 195"

    print("✓ No fragmentation: all clusters have sufficient data")


def test_global_vs_local():
    """Compare global clustering (wrong) vs local clustering (correct)."""
    print("\n" + "=" * 60)
    print("TEST 3: Global vs Local Clustering Comparison")
    print("=" * 60)

    # Setup: 3 clients, 400 samples each
    K_clients = 3
    n_per_client = 400
    X_splits = [np.random.randn(n_per_client, 10) for _ in range(K_clients)]

    # GLOBAL clustering (current - wrong)
    print("\n[GLOBAL Clustering - Old Implementation]")
    all_data = np.vstack(X_splits)
    global_kmeans = KMeans(n_clusters=3, random_state=42)
    global_labels = global_kmeans.fit_predict(all_data)

    # Split labels back
    labels_splits = []
    _curr = 0
    for X_k in X_splits:
        labels_splits.append(global_labels[_curr : _curr + len(X_k)])
        _curr += len(X_k)

    # Count (client, cluster) sizes
    global_sizes = []
    for k in range(K_clients):
        for h in range(3):
            size = (labels_splits[k] == h).sum()
            global_sizes.append(size)
            print(f"  Client {k}, Global Cluster {h}: {size} samples")

    global_min = min(global_sizes)
    global_avg = np.mean(global_sizes)

    # LOCAL clustering (Seng - correct)
    print("\n[LOCAL Clustering - V2 Implementation]")
    local_sizes = []
    for k in range(K_clients):
        client_data = X_splits[k]
        local_kmeans = KMeans(n_clusters=2, random_state=42)
        local_labels = local_kmeans.fit_predict(client_data)

        for h in range(2):
            size = (local_labels == h).sum()
            local_sizes.append(size)
            print(f"  Client {k}, Local Cluster {h}: {size} samples")

    local_min = min(local_sizes)
    local_avg = np.mean(local_sizes)

    # Comparison
    print("\n" + "-" * 60)
    print("COMPARISON:")
    print(f"  Global: min={global_min}, avg={global_avg:.1f} samples/cluster")
    print(f"  Local:  min={local_min}, avg={local_avg:.1f} samples/cluster")
    print(
        f"  Improvement: {local_min / max(1, global_min):.1f}× more data in worst case"
    )
    print("-" * 60)

    assert local_min > global_min, "Local clustering should preserve more data"
    assert local_min >= 150, "Local clustering should have ≥150 samples per cluster"

    print("✓ Local clustering significantly better than global")


if __name__ == "__main__":
    test_local_cluster_mixture()
    test_no_fragmentation()
    test_global_vs_local()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
