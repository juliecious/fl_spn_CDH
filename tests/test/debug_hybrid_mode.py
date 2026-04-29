"""
Debug script to investigate hybrid mode F1=0.000 bug.

Goal: Understand why hybrid mode fails in V2 while horizontal/vertical work.
"""

import logging
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import numpy as np
import torch
from argparse import Namespace

from causallearn.search.FCMBased.FedCDH import FedCDH
from causallearn.utils.data_utils import (
    my_simulate_linear_gaussian,
    set_random_seed,
    simulate_dag,
    simulate_parameter,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Configuration (minimal SMALL config)
set_random_seed(42)
d = 8
K = 3
n_total = 900
n_per_client = n_total // K  # 300 per client

# Generate data
logging.info("=== Generating synthetic data ===")
B_bin = simulate_dag(d, s0=d, graph_type="ER")
W = simulate_parameter(B_bin)
X_global, c_indx = my_simulate_linear_gaussian(W, n_total, standardize=True)

logging.info(f"Generated: d={d}, n_total={n_total}, K={K}")
logging.info(f"True DAG edges: {np.sum(B_bin)}")

# Split data for hybrid mode
X_splits = []
for k in range(K):
    start = k * n_per_client
    end = (k + 1) * n_per_client
    X_splits.append(X_global[start:end])

logging.info(f"\n=== Hybrid Mode Data Splits ===")
for k, xk in enumerate(X_splits):
    logging.info(f"Client {k}: shape={xk.shape}")

# Create FedCDH instance for hybrid mode
args = Namespace(
    data_type="linear",
    num_sums=20,
    num_leaves=20,
    num_repetitions=10,
    epochs=50,
    lr_spn=1e-3,
    alpha=0.05,
    indep_test="spn",
    device="cpu",
    num_local_clusters=2,  # V2 local clustering
    skip_spn_eval=True,  # Skip expensive evaluation
)

logging.info("\n=== Training Hybrid Mode SPN ===")
fedcdh = FedCDH(args)
fedcdh.scenario = "hybrid"
fedcdh.ci_method = "spn"
fedcdh.K_clients = K
fedcdh.d_features = d

# Just train the SPN, don't do full causal discovery yet
# We want to debug the SPN training and CI test behavior

# Manually reconstruct the training flow
X_aug_global = np.concatenate([X_global, c_indx], axis=1)
X_splits_aug = []
for k in range(K):
    start = k * n_per_client
    end = (k + 1) * n_per_client
    X_splits_aug.append(X_aug_global[start:end])

logging.info("\n=== Training FedSPN Model ===")
# This is a simplified version of the training logic from FedCDH.fit()
# We'll trace through to see where hybrid mode fails

from causallearn.utils.FedPC import (
    LocalSPNWrapper,
    LocalClusterMixture,
    build_feature_indicator_matrix,
    group_features_by_client_set,
    GroupMixture,
    ProductOverGroupsWithOverlap,
    compute_adaptive_hyperparameters,
)

from sklearn.cluster import KMeans

# Step 1: Train client local mixtures (same for all modes)
K_local = 2
client_local_mixtures = []

for k in range(K):
    logging.info(f"\n--- Client {k} ---")
    client_data = X_splits_aug[k]
    n_k = len(client_data)
    d_k = client_data.shape[1]

    logging.info(f"n_k={n_k}, d_k={d_k}")

    # LOCAL K-means
    kmeans = KMeans(n_clusters=K_local, random_state=42, n_init=10)
    local_cluster_labels = kmeans.fit_predict(client_data)

    cluster_spns = []
    cluster_weights = []

    for h in range(K_local):
        mask = local_cluster_labels == h
        cluster_data = client_data[mask]
        cluster_size = len(cluster_data)

        if cluster_size < 5:
            continue

        logging.info(f"  Cluster {h}: {cluster_size} samples")

        local_d = cluster_data.shape[1]
        hyperparams = compute_adaptive_hyperparameters(
            mode="hybrid",
            num_features=local_d,
            num_samples=cluster_size,
            data_type="linear",
            base_num_sums=20,
            base_num_leaves=20,
            base_epochs=50,
        )

        spn_kh = LocalSPNWrapper(
            num_features=local_d,
            device="cpu",
            num_sums=hyperparams["num_sums"],
            num_leaves=hyperparams["num_leaves"],
            depth=hyperparams["depth"],
            num_repetitions=10,
            seed=k * 10 + h,
        )
        spn_kh.train_local(
            cluster_data,
            epochs=hyperparams["epochs"],
            lr=1e-3,
            l1_weight=1e-4,
            l2_weight=hyperparams["weight_decay"],
            dropout=hyperparams["dropout"],
        )

        cluster_spns.append(spn_kh)
        cluster_weights.append(cluster_size)

    cluster_weights = np.array(cluster_weights) / n_k

    local_mixture = LocalClusterMixture(
        cluster_spns=cluster_spns,
        cluster_weights=cluster_weights,
        client_id=k,
        device="cpu",
    )
    client_local_mixtures.append(local_mixture)

    logging.info(
        f"  ✓ Local mixture: {len(cluster_spns)} clusters, weights={cluster_weights}"
    )

# Step 2: Build hybrid mode product (THIS IS WHERE THE BUG LIKELY IS)
logging.info("\n=== Building Hybrid Product ===")

# Build indicator matrix
M, feature_names = build_feature_indicator_matrix(
    X_splits=X_splits_aug, scenario="hybrid", d_features=d
)

logging.info(f"Indicator matrix shape: {M.shape}")
logging.info(f"Feature names: {feature_names}")

# Group features by client set
feature_subspaces = group_features_by_client_set(M, feature_names)
logging.info(f"Feature subspaces: {feature_subspaces}")

# Train SPNs per feature subspace
group_mixtures = []
feature_groups = []

for client_set, features in feature_subspaces.items():
    logging.info(
        f"\n--- Training SPNs for features {features} (clients {client_set}) ---"
    )

    trained_spns = []
    client_counts = []

    for k in client_set:
        client_data = X_splits_aug[k]
        client_data_subspace = client_data[:, features]
        local_d = client_data_subspace.shape[1]
        n_samples = len(client_data_subspace)

        logging.info(f"  Client {k}: {n_samples} samples × {local_d} features")

        hyperparams = compute_adaptive_hyperparameters(
            mode="hybrid",
            num_features=local_d,
            num_samples=n_samples,
            data_type="linear",
            base_num_sums=20,
            base_num_leaves=20,
            base_epochs=50,
        )

        spn_subspace = LocalSPNWrapper(
            num_features=local_d,
            device="cpu",
            num_sums=hyperparams["num_sums"],
            num_leaves=hyperparams["num_leaves"],
            depth=hyperparams["depth"],
            num_repetitions=10,
            seed=k * 1000 + hash(tuple(features)) % 1000,
        )
        spn_subspace.train_local(
            client_data_subspace,
            epochs=hyperparams["epochs"],
            lr=1e-3,
            l1_weight=1e-4,
            l2_weight=hyperparams["weight_decay"],
            dropout=hyperparams["dropout"],
        )

        trained_spns.append(spn_subspace)
        client_counts.append(n_samples)

    group_weights = np.array(client_counts) / np.array(client_counts).sum()

    group_mix = GroupMixture(
        client_spns=trained_spns,
        weights=group_weights.tolist(),
        feature_indices=features,
        device="cpu",
    )
    group_mixtures.append(group_mix)
    feature_groups.append(features)

    logging.info(
        f"  ✓ Feature group {features}: {len(trained_spns)} SPNs, weights={group_weights}"
    )

fed_spn = ProductOverGroupsWithOverlap(
    group_mixtures=group_mixtures,
    feature_groups=feature_groups,
    device="cpu",
    allow_overlap=True,
)

logging.info(f"\n✓ Hybrid product built: {len(group_mixtures)} feature groups")

# Step 3: Test CI queries
logging.info("\n=== Testing CI Queries ===")

# Create test query: X=0, Y=1, Z=[] (unconditional independence)
X_test = torch.tensor(X_aug_global, dtype=torch.float32)

# Test 1: Full data (no masking)
logging.info("\n--- Test 1: Full data query ---")
with torch.no_grad():
    ll_full = fed_spn.log_prob(X_test)
    logging.info(
        f"log P(X): mean={ll_full.mean().item():.3f}, std={ll_full.std().item():.3f}"
    )
    logging.info(
        f"  Non-finite values: {torch.isnan(ll_full).sum().item()} NaN, {torch.isinf(ll_full).sum().item()} Inf"
    )

# Test 2: Masked query (feature 0 observed, rest NaN)
logging.info("\n--- Test 2: Masked query (X0 observed) ---")
masked_batch = torch.full_like(X_test, float("nan"))
masked_batch[:, 0] = X_test[:, 0]  # Only feature 0 observed

with torch.no_grad():
    ll_masked = fed_spn.log_prob(masked_batch)
    logging.info(
        f"log P(X0): mean={ll_masked.mean().item():.3f}, std={ll_masked.std().item():.3f}"
    )
    logging.info(
        f"  Non-finite values: {torch.isnan(ll_masked).sum().item()} NaN, {torch.isinf(ll_masked).sum().item()} Inf"
    )

# Test 3: Query with features [0, 1]
logging.info("\n--- Test 3: Masked query (X0, X1 observed) ---")
masked_batch2 = torch.full_like(X_test, float("nan"))
masked_batch2[:, [0, 1]] = X_test[:, [0, 1]]

with torch.no_grad():
    ll_masked2 = fed_spn.log_prob(masked_batch2)
    logging.info(
        f"log P(X0, X1): mean={ll_masked2.mean().item():.3f}, std={ll_masked2.std().item():.3f}"
    )
    logging.info(
        f"  Non-finite values: {torch.isnan(ll_masked2).sum().item()} NaN, {torch.isinf(ll_masked2).sum().item()} Inf"
    )

# Test 4: Compare with horizontal mode (for reference)
logging.info("\n=== Comparison: Horizontal Mode ===")
from causallearn.utils.FedPC import GlobalFedSPN

# Build horizontal global mixture
dataset_weights = np.array([len(X_splits_aug[k]) for k in range(K)])
dataset_weights = dataset_weights / dataset_weights.sum()

fed_spn_horizontal = GlobalFedSPN(
    components=client_local_mixtures,
    weights=dataset_weights.tolist(),
    strategy="mixture",
    device="cpu",
)

with torch.no_grad():
    ll_horiz_full = fed_spn_horizontal.log_prob(X_test)
    logging.info(
        f"Horizontal log P(X): mean={ll_horiz_full.mean().item():.3f}, std={ll_horiz_full.std().item():.3f}"
    )
    logging.info(
        f"  Non-finite values: {torch.isnan(ll_horiz_full).sum().item()} NaN, {torch.isinf(ll_horiz_full).sum().item()} Inf"
    )

logging.info("\n=== Debug Complete ===")
logging.info("If hybrid log-likelihoods are NaN or -inf, that explains F1=0.000")
