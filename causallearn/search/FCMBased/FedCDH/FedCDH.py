import time
import numpy as np
import torch
import torch.nn as nn
import logging
import os
from typing import Dict, Any, Optional

# Enable MPS fallback for unsupported 5D operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.FedPC import (
    GlobalFedSPN,
    LocalSPNWrapper,
    LocalClusterMixture,
    UnivariateSPNWrapper,
    FederatedProduct,
    FederatedProductWithClusters,
    FederatedStructureLearner,
    GroupMixture,
    ProductOverGroups,
    ProductOverGroupsWithOverlap,
    build_feature_indicator_matrix,
    group_features_by_client_set,
    compute_adaptive_hyperparameters,
)
from causallearn.utils.fedpc_auto_structure import construct_fedpc_automatic
from causallearn.utils.data_utils import (
    count_dag_accuracy,
    count_skeleton_accuracy,
    get_cpdag_from_cdnod,
    get_dag_from_pdag,
    set_random_seed,
)
from causallearn.search.FCMBased.FedCDH.cost_analysis import (
    estimate_fedcdh_comm_cost,
    estimate_kci_comm_cost,
)


class SimulatedFederatedKMeans:
    """
    Simulates Federated K-Means to estimate Communication Cost and respect Privacy.
    Handles both Row-Partitioned (Horizontal/Hybrid) and Column-Partitioned (Vertical) data.
    """

    def __init__(self, n_clusters, max_iter=10, tol=1e-4, seed=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.comm_cost = 0  # Bytes
        self.labels_ = None
        self.inertia_ = 0

    def fit(self, X_splits, feature_maps, scenario):
        np.random.seed(self.seed)
        K_clients = len(X_splits)

        if scenario == "vertical":
            # Vertical: need feature_maps to determine total dimension
            N = X_splits[0].shape[0]
            D = 0
            for k in feature_maps:
                D = max(D, max(feature_maps[k]) + 1)
        elif scenario == "hybrid" and feature_maps is not None:
            # Hybrid: clients have different samples AND features
            # Determine total dimensions from feature_maps
            N = 0
            D = 0
            for k in range(K_clients):
                N = max(N, X_splits[k].shape[0])  # May have overlaps
                if k in feature_maps:
                    D = max(D, max(feature_maps[k]) + 1)
        else:
            # Horizontal: all clients see all features
            N = sum(len(x) for x in X_splits)
            D = X_splits[0].shape[1]

        if scenario == "vertical" or (
            scenario == "hybrid" and feature_maps is not None
        ):
            # Vertical/Hybrid: clients have different features, aggregate min/max
            g_min, g_max = np.full(D, np.inf), np.full(D, -np.inf)
            for k in range(K_clients):
                cols = feature_maps[k]
                l_min, l_max = X_splits[k].min(axis=0), X_splits[k].max(axis=0)
                g_min[cols] = np.minimum(g_min[cols], l_min)
                g_max[cols] = np.maximum(g_max[cols], l_max)
            # Fill any remaining dimensions (shouldn't happen if coverage is complete)
            g_min[np.isinf(g_min)] = 0.0
            g_max[np.isinf(g_max)] = 1.0
        else:
            # Horizontal: all clients have all features
            x0 = X_splits[0]
            if len(x0) > 0:
                g_min, g_max = x0.min(axis=0), x0.max(axis=0)
            else:
                g_min, g_max = np.zeros(D), np.ones(D)

        centroids = np.random.uniform(g_min, g_max, (self.n_clusters, D))

        for it in range(self.max_iter):
            prev_centroids = centroids.copy()

            if scenario == "hybrid" and feature_maps is not None:
                # HYBRID MODE: Overlapping samples and features
                # Each client has subset of samples and subset of features
                # Strategy: Compute distances using only features client has

                # Build global sample-to-labels mapping
                # Track which samples are seen by which clients
                sample_labels = {}  # sample_idx -> cluster_label
                sample_dists = {}  # sample_idx -> min_distance

                for k in range(K_clients):
                    if len(X_splits[k]) == 0:
                        continue

                    cols = feature_maps[k]  # Global feature indices
                    # X_splits[k] has local columns (0, 1, ..., d_k-1)
                    # centroids[:, cols] gets the relevant global features

                    # Compute distances for this client's samples
                    # X_splits[k]: (n_k, d_k) local features
                    # centroids[:, cols]: (n_clusters, d_k) relevant global features
                    dists = np.sum(
                        (X_splits[k][:, None, :] - centroids[None, :, cols]) ** 2,
                        axis=2,
                    )  # (n_k, n_clusters)

                    # Assign labels based on minimum distance
                    local_labels = np.argmin(dists, axis=1)
                    local_min_dists = np.min(dists, axis=1)

                    # Map to global sample indices (would need sample_maps)
                    # For now, accumulate all labels
                    # This is simplified - proper implementation needs sample_maps

                # Fallback to horizontal-style clustering for now
                # TODO: Implement proper sample_maps tracking
                global_sums = np.zeros((self.n_clusters, D))
                global_counts = np.zeros(self.n_clusters)
                self.inertia_ = 0
                all_labels = []

                for k in range(K_clients):
                    if len(X_splits[k]) == 0:
                        all_labels.append(np.array([]))
                        continue

                    cols = feature_maps[k]
                    # Compute distances using only features this client has
                    dists = np.sum(
                        (X_splits[k][:, None, :] - centroids[None, :, cols]) ** 2,
                        axis=2,
                    )

                    lbs = np.argmin(dists, axis=1)
                    all_labels.append(lbs)
                    self.inertia_ += np.min(dists, axis=1).sum()

                    # Update global sums - only for features this client has
                    for c in range(self.n_clusters):
                        mask = lbs == c
                        count = mask.sum()
                        if count > 0:
                            # X_splits[k][mask] has shape (count, d_k)
                            # We need to add to global_sums[c, cols]
                            global_sums[c, cols] += X_splits[k][mask].sum(axis=0)
                            global_counts[c] += count

                self.labels_ = (
                    np.concatenate(all_labels) if all_labels else np.array([])
                )

                # Update centroids
                new_centroids = np.zeros_like(centroids)
                for c in range(self.n_clusters):
                    if global_counts[c] > 0:
                        # Only update features that have data
                        for k in range(K_clients):
                            cols = feature_maps[k]
                            # Count how many samples from this client are in cluster c
                            # This is approximate - proper version needs sample tracking
                            new_centroids[c, cols] = (
                                global_sums[c, cols] / global_counts[c]
                            )
                    else:
                        new_centroids[c] = centroids[c]

            elif scenario == "vertical":
                for k in range(K_clients):
                    d_k = len(feature_maps[k])
                    self.comm_cost += self.n_clusters * d_k * 4
                partial_dists = np.zeros((N, self.n_clusters))
                for k in range(K_clients):
                    cols = feature_maps[k]
                    dist_k = np.sum(
                        (X_splits[k][:, None, :] - centroids[None, :, cols]) ** 2,
                        axis=2,
                    )
                    partial_dists += dist_k
                self.comm_cost += K_clients * N * self.n_clusters * 4
                self.labels_ = np.argmin(partial_dists, axis=1)
                self.inertia_ = np.min(partial_dists, axis=1).sum()
                self.comm_cost += K_clients * N * 4
                new_centroids = np.zeros_like(centroids)
                global_counts = np.zeros(self.n_clusters)
                for c in range(self.n_clusters):
                    global_counts[c] = (self.labels_ == c).sum()
                for k in range(K_clients):
                    cols = feature_maps[k]
                    for c in range(self.n_clusters):
                        mask = self.labels_ == c
                        if global_counts[c] > 0:
                            partial_sum = X_splits[k][mask].sum(axis=0)
                            new_centroids[c, cols] = partial_sum / global_counts[c]
                self.comm_cost += self.n_clusters * len(cols) * 4
            else:
                # HORIZONTAL MODE: All clients have same features, different samples
                global_sums = np.zeros((self.n_clusters, D))
                global_counts = np.zeros(self.n_clusters)
                self.inertia_ = 0
                all_labels = []
                self.comm_cost += K_clients * self.n_clusters * D * 4
                for k in range(K_clients):
                    if len(X_splits[k]) == 0:
                        all_labels.append(np.array([]))
                        continue

                    # All clients have same features (full D dimensions)
                    dists = (
                        np.linalg.norm(
                            X_splits[k][:, None, :] - centroids[None, :, :], axis=2
                        )
                        ** 2
                    )

                    lbs = np.argmin(dists, axis=1)
                    all_labels.append(lbs)
                    self.inertia_ += np.min(dists, axis=1).sum()

                    for c in range(self.n_clusters):
                        mask = lbs == c
                        count = mask.sum()
                        if count > 0:
                            global_sums[c] += X_splits[k][mask].sum(axis=0)
                            global_counts[c] += count

                self.labels_ = (
                    np.concatenate(all_labels) if all_labels else np.array([])
                )
                self.comm_cost += K_clients * self.n_clusters * (D + 1) * 4
                new_centroids = np.zeros_like(centroids)
                for c in range(self.n_clusters):
                    if global_counts[c] > 0:
                        new_centroids[c] = global_sums[c] / global_counts[c]
                    else:
                        new_centroids[c] = centroids[c]
            if np.linalg.norm(new_centroids - prev_centroids) < self.tol:
                break
            centroids = new_centroids
        return self


class FedCDH_SPN_Wrapper(torch.nn.Module):
    def __init__(self, global_spn: GlobalFedSPN, u_index: int, routing: bool = True):
        super().__init__()
        self.spn = global_spn
        self.u_index = u_index
        self.device = global_spn.device
        self.routing = routing

    def log_prob(self, x):
        if not self.routing:
            return self.spn.log_prob(x)

        # Context variable U is always the last column by convention
        # Separate features from context
        x_feat = x[:, :-1]
        u_col = x[:, -1]

        u_is_observed = not torch.isnan(u_col[0]).item()
        if u_is_observed:
            # When U is observed, we condition on it: p(x|U=k), NOT w_k * p(x|U=k)
            client_indices = u_col.long()
            unique_clients = torch.unique(client_indices)
            final_ll = torch.zeros(x.shape[0], 1, device=self.device)
            for k in unique_clients:
                k_idx = k.item()
                mask = client_indices == k
                x_sub = x_feat[mask]
                # Route to component k and return its log probability
                # No weight multiplication - we're conditioning, not marginalizing
                ll_sub = self.spn.log_prob_conditional_u(x_sub, k_idx)
                final_ll[mask] = ll_sub
            return final_ll
        else:
            return self.spn.log_prob(x_feat)

    def sample(self, n_samples):
        """
        Sample from the underlying SPN.
        Delegates to the wrapped GlobalFedSPN.sample() method.

        Args:
            n_samples: Number of samples to generate

        Returns:
            samples: Tensor of shape (n_samples, d_aug)
        """
        return self.spn.sample(n_samples)


class QueryCounterCIT:
    """
    Wrapper for CI test instances that counts the number of queries.
    Explicitly delegates common attributes instead of using __getattr__ magic.
    """

    def __init__(self, cit_instance):
        self.cit = cit_instance
        self.query_count = 0
        # Explicitly expose commonly used attributes
        self.method = getattr(cit_instance, "method", "unknown")
        self.data = getattr(cit_instance, "data", None)
        self.global_model = getattr(cit_instance, "global_model", None)

    def __call__(self, *args, **kwargs):
        self.query_count += 1
        if self.method in ["spn", "kci"]:
            return self.cit(*args, **kwargs)
        else:
            # FisherZ only needs first 3 args (X, Y, conditioning_set)
            return self.cit(*args[:3])


class FedCDH:
    def __init__(self, args: Dict[str, Any], sample_maps=None, feature_maps=None):
        self.args = args
        self.sample_maps = sample_maps
        self.feature_maps = feature_maps

        # Device selection with optional override
        # Priority: args.device > auto-detect (CUDA > CPU)
        # Note: MPS (Apple Silicon) disabled - simple-einet has issues with 5D tensor reductions
        if hasattr(args, "device") and args.device is not None:
            device_str = (
                args.device.lower()
                if isinstance(args.device, str)
                else str(args.device)
            )
            if device_str == "cuda":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                else:
                    logging.warning(
                        "CUDA requested but not available. Falling back to CPU."
                    )
                    self.device = torch.device("cpu")
            elif device_str == "cpu":
                self.device = torch.device("cpu")
            else:
                logging.warning(
                    f"Unknown device '{args.device}'. Using auto-detection."
                )
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            # Auto-detect: CUDA > CPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"FedCDH Initialized on device: {self.device}")
        self.K_clients = args.K
        self.d_features = args.d
        self.scenario = args.scenario
        self.model_type = args.model_type
        self.ci_method = args.ci_method
        self.n_samples_per_client = args.n
        self.fed_spn_model = None

        # Data type for adaptive hyperparameters (linear/nonlinear)
        # Default to "nonlinear" for conservative capacity estimates
        self.data_type = getattr(args, "data_type", "nonlinear")

        # Store local SPNs for post-hoc evaluation (tests/benchmarks/evaluate_spn.py)
        # Purpose: Enables SPN quality assessment (log-likelihood, MMD, KS tests)
        # to validate that local models learn correct distributions before causal discovery
        self.local_spns = []

        # V3: Structure-preserving aggregation for horizontal mode
        # Options: "mixture" (default), "structure_voting", "ll_weighted"
        self.horizontal_aggregation = getattr(args, "horizontal_aggregation", "mixture")
        self.structure_vote_threshold = getattr(args, "structure_vote_threshold", 0.5)

        # Gap 3: Automatic structure learning (Seng's Algorithm 1)
        # If True, automatically detect scenario from feature_maps
        self.auto_structure = getattr(args, "auto_structure", False)

        # Gap 4: Cluster-conditional vertical federation (Seng's Assumption 2)
        # If True, use FederatedProductWithClusters instead of naive FederatedProduct
        self.use_cluster_conditional = getattr(args, "use_cluster_conditional", False)

    def _extract_feature_indices(self, client_id: int, include_context: bool = False):
        """
        Extract feature indices for a client in vertical mode.

        Args:
            client_id: Client index
            include_context: If True, include context column index if present

        Returns:
            List of feature indices for the client
        """
        if not hasattr(self, "vertical_feature_map") or not self.vertical_feature_map:
            # Fallback: return all features
            return list(range(self.d_features))

        indices = self.vertical_feature_map.get(client_id, [])

        if not include_context:
            # Filter out context column (indices >= d_features)
            indices = [idx for idx in indices if idx < self.d_features]

        return indices

    def derive_covariances_from_spn(
        self, fed_spn_model, n_samples=5000, n_fourier_features=10, seed=42
    ):
        """
        Derive covariance tensor from trained SPN for FICP orientation.

        This is the KEY innovation: Instead of computing parallel summary statistics,
        we derive covariances FROM the SPN that was trained for skeleton discovery.

        Args:
            fed_spn_model: Trained GlobalFedSPN model
            n_samples: Number of samples to draw from SPN
            n_fourier_features: Number of random Fourier features (h)
            seed: Random seed for reproducibility

        Returns:
            covariance_tensor: Shape (d+1, d+1, h, h) where d is number of variables
                              Last dimension is domain/client indicator variable
        """
        logging.info(f"\n{'='*60}")
        logging.info(f"Deriving Covariances from Trained SPN")
        logging.info(f"  Samples: {n_samples}, Fourier features: {n_fourier_features}")
        logging.info(f"{'='*60}")

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Extract the underlying GlobalFedSPN from wrapper
        if hasattr(fed_spn_model, "model"):
            global_spn = fed_spn_model.model
        else:
            global_spn = fed_spn_model

        # Step 1: Sample from SPN
        logging.info(f"[Step 1/4] Sampling {n_samples} points from trained SPN...")
        with torch.no_grad():
            samples = global_spn.sample(n_samples)

        if torch.is_tensor(samples):
            samples = samples.cpu().numpy()

        n_vars = samples.shape[1]
        logging.info(f"  Sample shape: {samples.shape}")
        logging.info(
            f"  Sample statistics: mean={samples.mean(axis=0)[:3]}, std={samples.std(axis=0)[:3]}"
        )

        # Step 2: Initialize covariance tensor
        # Shape: (n_vars+1, n_vars+1, h, h) where last var is domain indicator
        logging.info(
            f"[Step 2/4] Initializing covariance tensor (shape: {n_vars+1} x {n_vars+1} x {n_fourier_features} x {n_fourier_features})..."
        )
        CT = np.zeros((n_vars + 1, n_vars + 1, n_fourier_features, n_fourier_features))

        # Step 3: Compute random Fourier features for each variable
        logging.info(f"[Step 3/4] Computing random Fourier features...")

        # Random Fourier feature parameters (shared across all variables)
        rff_w = np.random.randn(n_fourier_features)
        rff_b = np.random.uniform(0, 2 * np.pi, n_fourier_features)

        def compute_rff(x, h):
            """Compute random Fourier features for Gaussian kernel."""
            x = x.reshape(-1, 1)
            features = np.sqrt(2 / h) * np.cos(x @ rff_w.reshape(1, -1) + rff_b)
            return features

        # Compute features for all variables
        phi_vars = []
        for i in range(n_vars):
            phi = compute_rff(samples[:, i], n_fourier_features)
            phi_vars.append(phi)

        # Domain variable: Create synthetic domain labels for heterogeneity
        # In real federated setting, this would come from client indices
        # For now, assign domains based on sample order (simulate K domains)
        K_domains = self.K_clients
        samples_per_domain = n_samples // K_domains
        domain_labels = np.repeat(np.arange(K_domains), samples_per_domain)
        # Handle remainder
        if len(domain_labels) < n_samples:
            domain_labels = np.concatenate(
                [domain_labels, np.full(n_samples - len(domain_labels), K_domains - 1)]
            )

        # Domain features using delta kernel (one-hot encoding)
        phi_domain = np.zeros((n_samples, n_fourier_features))
        for idx in range(n_samples):
            phi_domain[idx, domain_labels[idx] % n_fourier_features] = 1.0

        # Step 4: Compute covariances
        logging.info(f"[Step 4/4] Computing pairwise covariances...")

        # Variable-variable covariances
        for i in range(n_vars):
            for j in range(n_vars):
                C_ij = (phi_vars[i].T @ phi_vars[j]) / n_samples
                CT[i, j, :, :] = C_ij

            # Variable-domain covariances
            C_i_domain = (phi_vars[i].T @ phi_domain) / n_samples
            CT[i, n_vars, :, :] = C_i_domain
            CT[n_vars, i, :, :] = C_i_domain.T

        # Domain self-covariance
        CT[n_vars, n_vars, :, :] = (phi_domain.T @ phi_domain) / n_samples

        # Log statistics
        non_zero = np.count_nonzero(CT)
        total_elements = CT.size
        tensor_norm = np.linalg.norm(CT)

        logging.info(f"\n{'='*60}")
        logging.info(f"Covariance Tensor Statistics:")
        logging.info(f"  Shape: {CT.shape}")
        logging.info(
            f"  Non-zero elements: {non_zero}/{total_elements} ({100*non_zero/total_elements:.1f}%)"
        )
        logging.info(f"  Frobenius norm: {tensor_norm:.4f}")
        logging.info(f"  Memory size: {CT.nbytes / 1024:.1f} KB")
        logging.info(f"{'='*60}\n")

        return CT

    def fit(self, X_splits, c_indx, true_DAG_bin):
        # Get training epochs and alpha from args
        if hasattr(self.args, "epochs"):
            train_epochs = self.args.epochs
        else:
            train_epochs = 50 if self.device.type in ["cuda", "gpu"] else 10
        alpha = self.args.alpha if hasattr(self.args, "alpha") else 0.05

        # Reconstruct global data from splits
        if isinstance(X_splits, list):
            if self.scenario == "vertical":
                # Vertical: concatenate features (axis=1)
                X_global = np.concatenate(X_splits, axis=1)
            elif self.scenario == "hybrid":
                # TRUE HYBRID: Reconstruct from overlapping splits
                from causallearn.search.FCMBased.FedCDH.data_partitioning.hybrid import (
                    reconstruct_from_hybrid_splits,
                )

                # If sample_maps not provided, infer from data splits and c_indx
                if self.sample_maps is None and c_indx is not None:
                    # Infer sample ownership from c_indx
                    sample_maps_inferred = {}
                    for k in range(len(X_splits)):
                        # Find which samples belong to client k
                        client_mask = c_indx.flatten() == k
                        sample_maps_inferred[k] = np.where(client_mask)[0]
                    self.sample_maps = sample_maps_inferred

                # Verify we have feature_maps for hybrid mode
                if self.feature_maps is None:
                    raise ValueError(
                        "Hybrid mode requires feature_maps. Please provide feature_maps to FedCDH constructor."
                    )

                X_global = reconstruct_from_hybrid_splits(
                    X_splits, self.sample_maps, self.feature_maps
                )
            else:
                # Horizontal: concatenate samples (axis=0)
                X_global = np.concatenate(X_splits, axis=0)
        else:
            X_global = X_splits

        total_samples = X_global.shape[0]

        # Augment with context column for CI testing
        X_aug_global = np.concatenate([X_global, c_indx], axis=1)
        d_aug_total = X_aug_global.shape[1]

        # Store augmented training data for evaluation (hybrid mode fix)
        self.X_aug_global_train = X_aug_global

        train_time = 0
        cd_time = 0
        comm_cost = 0.0
        clustering_cost = 0.0
        num_queries = 0

        if self.ci_method == "spn":
            start_train = time.time()

            # Feature maps needed for vertical and hybrid scenarios
            if self.scenario == "vertical":
                # Vertical: Build feature maps from X_splits (NO context column for training)
                # Context column only added to X_aug_global for CI testing
                if self.feature_maps is None:
                    feature_maps = {}
                    for k in range(self.K_clients):
                        # Feature indices are simply the columns in X_splits[k]
                        # Assuming X_splits are already properly partitioned
                        d_k = X_splits[k].shape[1]
                        # Map to global feature indices
                        start_idx = sum(X_splits[i].shape[1] for i in range(k))
                        feature_maps[k] = list(range(start_idx, start_idx + d_k))
                else:
                    feature_maps = self.feature_maps

                # Store original X_splits for training (WITHOUT context column)
                self.X_splits_train = X_splits
            elif self.scenario == "hybrid" and self.feature_maps is not None:
                # Hybrid: Use provided feature maps (with overlaps)
                feature_maps = self.feature_maps
                self.X_splits_train = X_splits
            else:
                # Horizontal: All clients see all features (no feature map needed)
                feature_maps = None
                if self.scenario == "horizontal":
                    # Keep existing splits but ensure they include context column U
                    new_splits = []
                    _curr = 0
                    for xk in X_splits:
                        new_splits.append(X_aug_global[_curr : _curr + len(xk)])
                        _curr += len(xk)
                    X_splits = new_splits
                else:  # hybrid
                    # BUGFIX: For hybrid mode, split X_global (WITHOUT context U)
                    # Context U will be added during CI testing phase, not SPN training
                    # This avoids dimension mismatch and double-counting issues
                    X_splits = np.array_split(X_global, self.K_clients)

            # Data partition validation
            logging.info(f"Data partition check: scenario={self.scenario}")
            total_samples_split = 0
            for k, xk in enumerate(X_splits):
                logging.info(f"  Client {k}: shape={xk.shape}")
                if self.scenario == "vertical":
                    # Vertical: all clients must have same number of samples
                    assert xk.shape[0] == X_aug_global.shape[0], (
                        f"Vertical scenario: Client {k} has {xk.shape[0]} samples, "
                        f"expected {X_aug_global.shape[0]} (all clients must see all samples)"
                    )
                else:
                    # Horizontal/Hybrid: samples are partitioned
                    total_samples_split += xk.shape[0]

            # Verify total samples match for horizontal/hybrid
            if self.scenario == "horizontal":
                assert total_samples_split == X_aug_global.shape[0], (
                    f"Horizontal scenario: Total samples across clients "
                    f"({total_samples_split}) != global samples ({X_aug_global.shape[0]})"
                )
            elif self.scenario == "hybrid":
                # Hybrid with overlaps: total may be larger due to overlap
                # Just check that global samples are covered
                if self.sample_maps is not None:
                    all_samples = set()
                    for smap in self.sample_maps.values():
                        all_samples.update(smap)
                    assert len(all_samples) >= X_aug_global.shape[0], (
                        f"Hybrid scenario: Only {len(all_samples)} unique samples covered, "
                        f"expected at least {X_aug_global.shape[0]}"
                    )
                else:
                    # Old hybrid mode (no overlaps)
                    assert total_samples_split == X_aug_global.shape[0], (
                        f"Hybrid scenario: Total samples across clients "
                        f"({total_samples_split}) != global samples ({X_aug_global.shape[0]})"
                    )

            logging.info(
                f"✓ Data partition validation passed for {self.scenario} scenario"
            )

            # V2 OPTION 1: FedCDH Baseline (Following Paper)
            # Use surrogate variable ℧ = client index (no k-means clustering)
            # Rationale: For homogeneous synthetic data, client index IS the mechanism
            # For heterogeneous real data, can enable use_kmeans_clustering=True
            #
            # Reference: Li et al. (2024) FedCDH ICLR 2024
            # "We utilize a surrogate variable corresponding to the client or domain index"

            use_kmeans = getattr(self.args, "use_kmeans_clustering", False)
            force_num_clusters = getattr(self.args, "force_num_clusters", None)

            if force_num_clusters is not None:
                # Override: force specific cluster count (for testing/comparison)
                num_clusters = force_num_clusters
                logging.info(f"[Override] Cluster count FORCED to K={num_clusters}")
                fed_km = SimulatedFederatedKMeans(
                    n_clusters=num_clusters, max_iter=10, seed=42
                )
                fed_km.fit(X_splits, feature_maps, self.scenario)
                best_model = fed_km

            elif not use_kmeans:
                # DEFAULT (Option 1): Use client index as surrogate variable
                # This matches FedCDH paper exactly - no mechanism discovery needed
                num_clusters = self.K_clients
                logging.info(
                    f"[FedCDH Baseline] Using K={num_clusters} clusters "
                    f"(surrogate ℧ = client index, following Li et al. 2024)"
                )

                # Simple assignment: cluster h = client k
                fed_km = SimulatedFederatedKMeans(
                    n_clusters=num_clusters, max_iter=10, seed=42
                )
                fed_km.fit(X_splits, feature_maps, self.scenario)
                best_model = fed_km

            else:
                # OPTIONAL (Option 2 - Future): K-means for mechanism discovery
                # Only use this for truly heterogeneous data with multiple mechanisms
                # Example: Real hospital data, different experimental conditions
                logging.info(
                    "[K-means Mode] Discovering mechanisms via BIC selection (experimental)"
                )

                best_h = 2
                min_bic = float("inf")
                best_model = None
                bic_scores = []
                for h_candidate in range(2, 6):
                    fed_km = SimulatedFederatedKMeans(
                        n_clusters=h_candidate, max_iter=10, seed=42
                    )
                    fed_km.fit(X_splits, feature_maps, self.scenario)
                    bic = (
                        fed_km.inertia_
                        + h_candidate * np.log(total_samples) * d_aug_total
                    )
                    bic_scores.append((h_candidate, bic))
                    if bic < min_bic:
                        min_bic = bic
                        best_h = h_candidate
                        best_model = fed_km

                # Data availability constraint
                min_samples_per_group = 100
                max_clusters_by_data = max(
                    2, total_samples // (self.K_clients * min_samples_per_group)
                )
                num_clusters = min(best_h, max_clusters_by_data)

                if num_clusters < best_h:
                    logging.info(
                        f"BIC selection: K={best_h} from {bic_scores}, "
                        f"capped to K={num_clusters} (need ≥{min_samples_per_group} samples/cluster/client)"
                    )
                else:
                    logging.info(
                        f"BIC selection: chosen K={num_clusters} from {bic_scores}"
                    )

            labels = best_model.labels_
            clustering_cost = best_model.comm_cost / 1024.0
            labels_splits = []
            _curr = 0
            for _xk in X_splits:
                labels_splits.append(labels[_curr : _curr + len(_xk)])
                _curr += len(_xk)
            weights = np.bincount(labels, minlength=num_clusters) / len(labels)
            clients_clusters = [[] for _ in range(num_clusters)]
            clients_counts = [[] for _ in range(num_clusters)]

            # Adaptive hyperparameters based on dimensionality
            # Rationale: Higher dimensions need more model capacity and training
            # Updated defaults: 5→20 for num_sums/leaves (4× capacity improvement)
            # Rationale: RAT-SPN literature uses 20-40 for similar problems
            num_sums = getattr(self.args, "num_sums", 20)
            num_leaves = getattr(self.args, "num_leaves", 20)
            num_repetitions = getattr(self.args, "num_repetitions", 10)

            # Ensemble configuration (adaptive strategy)
            # DEFAULT CHANGED: n_ensemble=1 for benchmarks (5× faster training)
            # Ensemble improves robustness but adds 5× training time
            # For production use, set args.n_ensemble=5 explicitly
            n_ensemble = getattr(self.args, "n_ensemble", None)
            if n_ensemble is None:
                # Default to single model for faster benchmarking
                # Old auto-detect logic (too slow for benchmarks):
                # if self.d_features >= 8 or self.scenario in ["vertical", "hybrid"]:
                #     n_ensemble = 5  # Ensemble for complex cases
                n_ensemble = 1  # Single model (5× faster than ensemble)
            logging.info(
                f"SPN configuration: n_ensemble={n_ensemble} "
                f"({'auto-detected (single model for speed)' if getattr(self.args, 'n_ensemble', None) is None else 'user-specified'})"
            )

            # Adaptive learning rate: decrease for higher dimensions
            # Formula: lr = 0.01 / sqrt(d/5)
            # Effect: d=5→0.010, d=8→0.008, d=10→0.007, d=15→0.006
            base_lr = getattr(self.args, "lr", 0.01)
            adaptive_lr = base_lr / np.sqrt(max(1.0, self.d_features / 5.0))
            logging.info(
                f"Adaptive learning rate: base={base_lr:.4f}, "
                f"d={self.d_features} → lr={adaptive_lr:.4f}"
            )

            # Adaptive epochs: scale with complexity
            # Formula: epochs = base_epochs * (d/5)^1.5
            # Effect: d=5→base, d=8→2.3×base, d=10→2.8×base
            base_epochs = train_epochs
            adaptive_epochs = int(base_epochs * (self.d_features / 5.0) ** 1.5)
            adaptive_epochs = max(base_epochs, adaptive_epochs)  # Never less than base
            if adaptive_epochs != base_epochs:
                logging.info(
                    f"Adaptive epochs: base={base_epochs}, "
                    f"d={self.d_features} → epochs={adaptive_epochs} ({adaptive_epochs/base_epochs:.1f}×)"
                )
            else:
                logging.info(f"Training epochs: {adaptive_epochs}")
            train_epochs = adaptive_epochs

            # === MODE-SPECIFIC CAPACITY ADJUSTMENTS ===
            # Vertical and Hybrid modes have fundamental architectural challenges
            # that require additional capacity beyond standard adaptive scaling
            logging.info(f"\n{'='*60}")
            logging.info(f"[MODE-SPECIFIC ADJUSTMENTS] Scenario: {self.scenario}")
            logging.info(f"{'='*60}")

            capacity_multiplier = 1.0
            epoch_multiplier = 1.0
            K_local_override = None

            if self.scenario == "vertical":
                # Vertical mode challenge: Feature fragmentation across clients
                # Each client sees d/K features, must learn cross-client dependencies
                d_per_client = self.d_features / self.K_clients

                logging.info(
                    f"  Vertical mode: d/K = {d_per_client:.2f} features per client"
                )

                # Critical fix: Disable clustering if insufficient features
                if d_per_client < 3:
                    K_local_override = 1
                    logging.info(
                        f"  → CRITICAL: d/K < 3, forcing K_local=1 (clustering disabled)"
                    )
                    logging.info(
                        f"     Reason: Cannot meaningfully cluster <3 features per client"
                    )

                # Adaptive capacity scaling inversely with features per client
                # Fewer features → need more capacity to compensate
                if d_per_client < 2:
                    capacity_multiplier = 3.0
                    epoch_multiplier = 4.0
                    logging.info(
                        f"  → Severe fragmentation: 3.0× capacity, 4.0× epochs"
                    )
                elif d_per_client < 4:
                    capacity_multiplier = 2.0
                    epoch_multiplier = 2.5
                    logging.info(f"  → High fragmentation: 2.0× capacity, 2.5× epochs")
                elif d_per_client < 8:
                    capacity_multiplier = 1.3
                    epoch_multiplier = 1.5
                    logging.info(
                        f"  → Moderate fragmentation: 1.3× capacity, 1.5× epochs"
                    )
                else:
                    capacity_multiplier = 1.0
                    epoch_multiplier = 1.2
                    logging.info(
                        f"  → Low fragmentation: baseline capacity, 1.2× epochs"
                    )

            elif self.scenario == "hybrid":
                # Hybrid mode challenge: Dual complexity
                # - Sample partitioning (like horizontal)
                # - Sum-of-Products aggregation (more complex than mixture)

                # Memory-aware adjustment: Small d causes OOM in Sum-of-Products
                # Reason: With small d, local clusters create large intermediate tensors
                if self.d_features < 15:
                    capacity_multiplier = 1.3  # Reduced from 1.6
                    epoch_multiplier = 1.5  # Reduced from 1.8
                    K_local_override = 1  # Disable clustering to save memory
                    logging.info(
                        f"  Hybrid mode (d={self.d_features} < 15): "
                        f"1.3× capacity, 1.5× epochs, K_local=1 (memory-aware)"
                    )
                    logging.info(
                        f"    → Reason: Small d with Sum-of-Products can cause CUDA OOM"
                    )
                else:
                    capacity_multiplier = 1.6
                    epoch_multiplier = 1.8
                    logging.info(
                        f"  Hybrid mode: 1.6× capacity, 1.8× epochs (dual challenge compensation)"
                    )

            else:  # horizontal
                logging.info(f"  Horizontal mode: baseline capacity (no adjustment)")

            logging.info(f"{'='*60}\n")

            # V2 LOCAL CLUSTERING: Seng et al. (2025) Algorithm 1
            # KEY CHANGE: Client-first loop (not cluster-first) with LOCAL k-means per client
            # Rationale: Prevents data fragmentation, preserves sample sufficiency
            # Reference: experiments/SPN_STRUCTURE_COMPARISON.md Section 1.1

            from sklearn.cluster import KMeans

            # Determine local cluster count
            K_local = getattr(self.args, "num_local_clusters", 2)

            # Apply mode-specific override if needed
            if K_local_override is not None:
                K_local = K_local_override
            # Safety: Ensure at least 100 samples per local cluster
            min_samples_per_cluster = 100
            for X_k in X_splits:
                max_K_local = max(1, len(X_k) // min_samples_per_cluster)
                K_local = min(K_local, max_K_local)
            K_local = max(1, K_local)  # At least 1 cluster

            logging.info(
                f"\n{'='*60}\n"
                f"[V2 LOCAL CLUSTERING] Following Seng et al. (2025) Algorithm 1\n"
                f"K_local={K_local} clusters per client (not global clustering)\n"
                f"{'='*60}"
            )

            # Store client local mixtures (one per client)
            client_local_mixtures = []

            # CRITICAL: Loop over clients FIRST (not clusters)
            for k in range(self.K_clients):
                logging.info(f"\n{'='*60}")
                logging.info(f"Training Client {k}/{self.K_clients}")
                logging.info(f"{'='*60}")

                # Get full client data (no fragmentation)
                client_data = X_splits[k]
                n_k = len(client_data)
                d_k = client_data.shape[1]

                logging.info(f"Client {k} data: n={n_k}, d={d_k}")

                # LOCAL clustering on this client's data
                if K_local > 1 and n_k >= 20:
                    logging.info(f"  Performing LOCAL K-means (K_local={K_local})...")

                    # K-means on LOCAL data (not global!)
                    # Force single-threaded to avoid OpenMP hang on macOS
                    import os

                    old_omp = os.environ.get("OMP_NUM_THREADS", None)
                    os.environ["OMP_NUM_THREADS"] = "1"
                    try:
                        kmeans = KMeans(n_clusters=K_local, random_state=42, n_init=10)
                        local_cluster_labels = kmeans.fit_predict(client_data)
                    finally:
                        # Restore original value
                        if old_omp is not None:
                            os.environ["OMP_NUM_THREADS"] = old_omp
                        else:
                            os.environ.pop("OMP_NUM_THREADS", None)

                    # Log cluster distribution
                    unique, counts = np.unique(local_cluster_labels, return_counts=True)
                    cluster_dist = dict(zip(unique.tolist(), counts.tolist()))
                    logging.info(f"  Local cluster distribution: {cluster_dist}")

                    # Train K_local SPNs, one per local cluster
                    cluster_spns = []
                    cluster_weights = []

                    for h in range(K_local):
                        mask = local_cluster_labels == h
                        cluster_data = client_data[mask]
                        cluster_size = len(cluster_data)

                        if cluster_size < 5:
                            logging.warning(
                                f"    Skipping local cluster {h} (insufficient data: {cluster_size} < 5)"
                            )
                            continue

                        logging.info(
                            f"  Training SPN for local cluster {h}: {cluster_size} samples"
                        )

                        local_d = cluster_data.shape[1]

                        # Get adaptive hyperparameters
                        hyperparams = compute_adaptive_hyperparameters(
                            mode=self.scenario,
                            num_features=local_d,
                            num_samples=cluster_size,
                            data_type=self.data_type,
                            base_num_sums=num_sums,
                            base_num_leaves=num_leaves,
                            base_epochs=train_epochs,
                        )

                        # Apply mode-specific capacity multipliers
                        hyperparams["num_sums"] = int(
                            hyperparams["num_sums"] * capacity_multiplier
                        )
                        hyperparams["num_leaves"] = int(
                            hyperparams["num_leaves"] * capacity_multiplier
                        )
                        hyperparams["epochs"] = int(
                            hyperparams["epochs"] * epoch_multiplier
                        )

                        # Create and train SPN for this local cluster with error handling
                        try:
                            if local_d == 1:
                                spn_kh = UnivariateSPNWrapper(
                                    device=self.device,
                                    num_sums=hyperparams["num_sums"],
                                    num_leaves=hyperparams["num_leaves"],
                                    seed=k * 10 + h,
                                )
                                spn_kh.train_local(
                                    cluster_data,
                                    epochs=hyperparams["epochs"],
                                    lr=adaptive_lr,
                                )
                            else:
                                spn_kh = LocalSPNWrapper(
                                    num_features=local_d,
                                    device=self.device,
                                    num_sums=hyperparams["num_sums"],
                                    num_leaves=hyperparams["num_leaves"],
                                    depth=hyperparams["depth"],
                                    num_repetitions=num_repetitions,
                                    seed=k * 10 + h,
                                )
                                spn_kh.train_local(
                                    cluster_data,
                                    epochs=hyperparams["epochs"],
                                    lr=adaptive_lr,
                                    l1_weight=1e-4,
                                    l2_weight=hyperparams["weight_decay"],
                                    dropout=hyperparams["dropout"],
                                )

                            cluster_spns.append(spn_kh)
                            cluster_weights.append(cluster_size)

                        except torch.cuda.OutOfMemoryError as oom_error:
                            logging.error(
                                f"    CUDA OOM during training cluster {h} for client {k}. "
                                f"Trying CPU fallback..."
                            )
                            # Clear CUDA cache
                            torch.cuda.empty_cache()

                            # Retry on CPU
                            try:
                                if local_d == 1:
                                    spn_kh = UnivariateSPNWrapper(
                                        device="cpu",
                                        num_sums=num_sums,
                                        num_leaves=num_leaves,
                                        seed=k * 10 + h,
                                    )
                                    spn_kh.train_local(
                                        cluster_data,
                                        epochs=hyperparams["epochs"],
                                        lr=adaptive_lr,
                                    )
                                else:
                                    spn_kh = LocalSPNWrapper(
                                        num_features=local_d,
                                        device="cpu",
                                        num_sums=hyperparams["num_sums"],
                                        num_leaves=hyperparams["num_leaves"],
                                        depth=hyperparams["depth"],
                                        num_repetitions=num_repetitions,
                                        seed=k * 10 + h,
                                    )
                                    spn_kh.train_local(
                                        cluster_data,
                                        epochs=hyperparams["epochs"],
                                        lr=adaptive_lr,
                                        l1_weight=1e-4,
                                        l2_weight=hyperparams["weight_decay"],
                                        dropout=hyperparams["dropout"],
                                    )

                                # Move trained model back to original device to avoid device mismatch
                                try:
                                    # Move main model
                                    spn_kh.model = spn_kh.model.to(self.device)

                                    # Move normalization tensors if they exist
                                    if spn_kh.mean is not None:
                                        spn_kh.mean = spn_kh.mean.to(self.device)
                                    if spn_kh.std is not None:
                                        spn_kh.std = spn_kh.std.to(self.device)

                                    # Move variable ordering tensors if they exist
                                    if (
                                        hasattr(spn_kh, "variable_order")
                                        and spn_kh.variable_order is not None
                                    ):
                                        spn_kh.variable_order = (
                                            spn_kh.variable_order.to(self.device)
                                        )
                                    if (
                                        hasattr(spn_kh, "inv_variable_order")
                                        and spn_kh.inv_variable_order is not None
                                    ):
                                        spn_kh.inv_variable_order = (
                                            spn_kh.inv_variable_order.to(self.device)
                                        )

                                    # Update device attribute
                                    spn_kh.device = self.device

                                    logging.info(
                                        f"    ✓ CPU fallback successful for cluster {h}, "
                                        f"all tensors moved to {self.device}"
                                    )
                                except Exception as move_error:
                                    logging.warning(
                                        f"    ✓ CPU fallback successful but could not move to {self.device}, "
                                        f"keeping on CPU: {move_error}"
                                    )
                                    # Keep on CPU - will handle in aggregation

                                cluster_spns.append(spn_kh)
                                cluster_weights.append(cluster_size)
                            except Exception as cpu_error:
                                logging.error(
                                    f"    CPU fallback also failed for cluster {h}: {cpu_error}"
                                )
                                logging.warning(
                                    f"    Skipping cluster {h} due to training failure"
                                )
                                continue

                        except Exception as e:
                            logging.error(f"    Training failed for cluster {h}: {e}")
                            logging.warning(f"    Skipping cluster {h}")
                            continue

                    # Normalize weights
                    if len(cluster_spns) > 0:
                        cluster_weights = np.array(cluster_weights) / n_k

                        # Build local mixture for this client
                        local_mixture = LocalClusterMixture(
                            cluster_spns=cluster_spns,
                            cluster_weights=cluster_weights,
                            client_id=k,
                            device=self.device,
                        )
                        client_local_mixtures.append(local_mixture)

                        logging.info(
                            f"  ✓ Client {k} local mixture: {len(cluster_spns)} clusters, "
                            f"weights={cluster_weights}"
                        )
                    else:
                        raise RuntimeError(
                            f"Client {k}: No valid local clusters created!"
                        )

                else:
                    # No clustering: single SPN for entire client data
                    logging.info(f"  Training single SPN (K_local=1, n={n_k})")

                    local_d = client_data.shape[1]

                    hyperparams = compute_adaptive_hyperparameters(
                        mode=self.scenario,
                        num_features=local_d,
                        num_samples=n_k,
                        data_type=self.data_type,
                        base_num_sums=num_sums,
                        base_num_leaves=num_leaves,
                        base_epochs=train_epochs,
                    )

                    # Apply mode-specific capacity multipliers
                    hyperparams["num_sums"] = int(
                        hyperparams["num_sums"] * capacity_multiplier
                    )
                    hyperparams["num_leaves"] = int(
                        hyperparams["num_leaves"] * capacity_multiplier
                    )
                    hyperparams["epochs"] = int(
                        hyperparams["epochs"] * epoch_multiplier
                    )

                    try:
                        if local_d == 1:
                            single_spn = UnivariateSPNWrapper(
                                device=self.device,
                                num_sums=hyperparams["num_sums"],
                                num_leaves=hyperparams["num_leaves"],
                                seed=k * 10,
                            )
                            single_spn.train_local(
                                client_data,
                                epochs=hyperparams["epochs"],
                                lr=adaptive_lr,
                            )
                        else:
                            single_spn = LocalSPNWrapper(
                                num_features=local_d,
                                device=self.device,
                                num_sums=hyperparams["num_sums"],
                                num_leaves=hyperparams["num_leaves"],
                                depth=hyperparams["depth"],
                                num_repetitions=num_repetitions,
                                seed=k * 10,
                            )
                            single_spn.train_local(
                                client_data,
                                epochs=hyperparams["epochs"],
                                lr=adaptive_lr,
                                l1_weight=1e-4,
                                l2_weight=hyperparams["weight_decay"],
                                dropout=hyperparams["dropout"],
                            )

                        # Wrap in LocalClusterMixture for consistency (K_local=1)
                        local_mixture = LocalClusterMixture(
                            cluster_spns=[single_spn],
                            cluster_weights=[1.0],
                            client_id=k,
                            device=self.device,
                        )
                        client_local_mixtures.append(local_mixture)

                        logging.info(f"  ✓ Client {k} single SPN trained")

                    except torch.cuda.OutOfMemoryError as oom_error:
                        logging.error(
                            f"  CUDA OOM during training client {k} single SPN. "
                            f"Trying CPU fallback..."
                        )
                        # Clear CUDA cache
                        torch.cuda.empty_cache()

                        # Retry on CPU
                        try:
                            if local_d == 1:
                                single_spn = UnivariateSPNWrapper(
                                    device="cpu",
                                    num_sums=num_sums,
                                    num_leaves=num_leaves,
                                    seed=k * 10,
                                )
                                single_spn.train_local(
                                    client_data,
                                    epochs=hyperparams["epochs"],
                                    lr=adaptive_lr,
                                )
                            else:
                                single_spn = LocalSPNWrapper(
                                    num_features=local_d,
                                    device="cpu",
                                    num_sums=hyperparams["num_sums"],
                                    num_leaves=hyperparams["num_leaves"],
                                    depth=hyperparams["depth"],
                                    num_repetitions=num_repetitions,
                                    seed=k * 10,
                                )
                                single_spn.train_local(
                                    client_data,
                                    epochs=hyperparams["epochs"],
                                    lr=adaptive_lr,
                                    l1_weight=1e-4,
                                    l2_weight=hyperparams["weight_decay"],
                                    dropout=hyperparams["dropout"],
                                )

                            # Move trained model back to original device to avoid device mismatch
                            try:
                                # Move main model
                                single_spn.model = single_spn.model.to(self.device)

                                # Move normalization tensors if they exist
                                if single_spn.mean is not None:
                                    single_spn.mean = single_spn.mean.to(self.device)
                                if single_spn.std is not None:
                                    single_spn.std = single_spn.std.to(self.device)

                                # Move variable ordering tensors if they exist
                                if (
                                    hasattr(single_spn, "variable_order")
                                    and single_spn.variable_order is not None
                                ):
                                    single_spn.variable_order = (
                                        single_spn.variable_order.to(self.device)
                                    )
                                if (
                                    hasattr(single_spn, "inv_variable_order")
                                    and single_spn.inv_variable_order is not None
                                ):
                                    single_spn.inv_variable_order = (
                                        single_spn.inv_variable_order.to(self.device)
                                    )

                                # Update device attribute
                                single_spn.device = self.device

                                logging.info(
                                    f"  ✓ CPU fallback successful for client {k}, "
                                    f"all tensors moved to {self.device}"
                                )
                            except Exception as move_error:
                                logging.warning(
                                    f"  ✓ CPU fallback successful but could not move to {self.device}, "
                                    f"keeping on CPU: {move_error}"
                                )
                                # Keep on CPU - will handle in aggregation

                            # Wrap in LocalClusterMixture for consistency (K_local=1)
                            local_mixture = LocalClusterMixture(
                                cluster_spns=[single_spn],
                                cluster_weights=[1.0],
                                client_id=k,
                                device=self.device,  # Use original device, not hardcoded "cpu"
                            )
                            client_local_mixtures.append(local_mixture)

                        except Exception as cpu_error:
                            logging.error(
                                f"  CPU fallback also failed for client {k}: {cpu_error}"
                            )
                            raise RuntimeError(
                                f"Client {k}: Failed to train single SPN on both CUDA and CPU. "
                                f"Cannot proceed without at least one successful client."
                            )

                    except Exception as e:
                        logging.error(
                            f"  Training failed for client {k} single SPN: {e}"
                        )
                        raise RuntimeError(
                            f"Client {k}: Failed to train single SPN. Cannot proceed."
                        )

            logging.info(f"\n{'='*60}")
            logging.info(
                f"All {self.K_clients} clients trained successfully (LOCAL clustering)"
            )
            logging.info(f"{'='*60}\n")

            # Store for backward compatibility
            # Note: Old code expected clients_clusters[h][k] = SPN
            # New code has client_local_mixtures[k] = LocalClusterMixture
            # We need to maintain compatibility with aggregation code below
            clients_clusters = [[]]  # Dummy structure
            clients_counts = [[]]

            # V2 GLOBAL AGGREGATION: Mode-specific strategies
            # Build global SPN from client local mixtures
            logging.info(f"\n{'='*60}")
            logging.info(
                f"[V2 GLOBAL AGGREGATION] Building global SPN for {self.scenario} mode"
            )
            logging.info(f"{'='*60}\n")

            global_components = []
            final_weights = []

            if self.scenario == "horizontal":
                # HORIZONTAL MODE: Mixture over client local mixtures
                # V3 Enhancement: Support structure-preserving aggregation
                logging.info(
                    "[Horizontal Mode] Building global mixture over client mixtures"
                )
                logging.info(f"  Aggregation method: {self.horizontal_aggregation}")

                # Dataset weights (proportional to sample counts)
                dataset_weights = np.array(
                    [len(X_splits[k]) for k in range(self.K_clients)]
                )
                dataset_weights = dataset_weights / dataset_weights.sum()

                logging.info(f"  Dataset weights: {dataset_weights}")

                # V3: Choose aggregation strategy
                if self.horizontal_aggregation == "structure_voting":
                    # Solution 1: Structure-preserving via majority voting
                    from causallearn.search.FCMBased.FedCDH.data_partitioning.aggregation import (
                        extract_local_dependency_graph,
                        aggregate_structures_by_voting,
                        log_structure_aggregation_summary,
                    )

                    logging.info(
                        "  [Structure Voting] Extracting dependency graphs from local SPNs..."
                    )

                    # Extract local dependency graphs
                    local_graphs = []
                    for k, (spn, X_k) in enumerate(
                        zip(client_local_mixtures, X_splits)
                    ):
                        logging.info(f"    Client {k}: Extracting dependencies...")
                        graph = extract_local_dependency_graph(
                            spn_model=spn,
                            X_data=X_k,
                            alpha=alpha,
                            num_permutations=50,
                            device=str(self.device),
                            has_augmented_var=True,  # Condition on augmented variable
                        )
                        local_graphs.append(graph)
                        logging.info(f"      → {len(graph.edges())} edges detected")

                    # Aggregate via majority voting
                    consensus_graph, edge_confidence = aggregate_structures_by_voting(
                        local_graphs=local_graphs,
                        threshold=self.structure_vote_threshold,
                    )

                    # Log summary
                    log_structure_aggregation_summary(
                        local_graphs=local_graphs,
                        consensus_graph=consensus_graph,
                        edge_confidence=edge_confidence,
                    )

                    # Store consensus for downstream causal discovery
                    # Note: This graph can be used to constrain PC algorithm or as direct output
                    self.consensus_dependency_graph = consensus_graph
                    self.edge_confidence = edge_confidence

                    # FIX #3: Convert consensus graph to initial skeleton for PC algorithm
                    # The consensus graph from structure voting provides high-quality edge proposals
                    # Use it as initial skeleton to speed up PC algorithm and improve accuracy
                    n_vars = self.d_features
                    initial_skeleton = np.zeros((n_vars, n_vars), dtype=int)
                    for edge in consensus_graph.edges():
                        i, j = edge
                        # Undirected skeleton: mark both directions
                        initial_skeleton[i, j] = 1
                        initial_skeleton[j, i] = 1

                    self.initial_skeleton_from_voting = initial_skeleton
                    logging.info(
                        f"  [Structure Voting] Converted consensus graph to initial skeleton: "
                        f"{initial_skeleton.sum() // 2} edges"
                    )

                    # Build standard mixture for SPN inference (still needed for CI tests)
                    fed_spn = GlobalFedSPN(
                        components=client_local_mixtures,
                        weights=dataset_weights.tolist(),
                        strategy="mixture",
                        device=self.device,
                    )

                    logging.info(
                        "  ✓ Structure voting completed + global mixture built"
                    )

                elif self.horizontal_aggregation == "ll_weighted":
                    # Solution 2: Confidence-weighted by log-likelihood
                    from causallearn.search.FCMBased.FedCDH.data_partitioning.aggregation import (
                        build_structure_weighted_mixture,
                    )

                    logging.info("  [LL Weighted] Computing quality-based weights...")

                    fed_spn, quality_weights = build_structure_weighted_mixture(
                        local_spns=client_local_mixtures,
                        local_data=X_splits,
                        device=str(self.device),
                        weight_by_ll=True,
                    )

                    logging.info(f"    Sample weights: {dataset_weights}")
                    logging.info(f"    Quality weights: {quality_weights}")
                    logging.info("  ✓ LL-weighted mixture built")

                else:
                    # Default: Standard mixture (V2 behavior)
                    # P(X) = Σ_k w_k × P_k(X)
                    fed_spn = GlobalFedSPN(
                        components=client_local_mixtures,
                        weights=dataset_weights.tolist(),
                        strategy="mixture",
                        device=self.device,
                    )

                    logging.info("  ✓ Horizontal global mixture built (default)")

            # === GAP 3: AUTOMATIC STRUCTURE LEARNING ===
            # If auto_structure=True, automatically detect scenario from feature_maps
            if self.auto_structure and self.feature_maps is not None:
                logging.info("\n[Automatic Structure Learning] Gap 3 Integration")
                fed_spn, detected_scenario = construct_fedpc_automatic(
                    local_spns=client_local_mixtures,
                    feature_maps=self.feature_maps,
                    num_features=self.d_features,
                    num_clusters=K_local,
                    device=self.device,
                    verbose=self.verbose,
                )
                logging.info(f"  ✓ Auto-detected scenario: {detected_scenario}")
                logging.info(f"  ✓ Built optimal structure: {type(fed_spn).__name__}")
                # Override scenario for consistency in downstream code
                self.scenario = detected_scenario

            elif self.scenario in ["vertical", "hybrid"]:
                # === GAP 4: CLUSTER-CONDITIONAL VERTICAL (Optional) ===
                # If use_cluster_conditional=True and scenario=vertical, use simpler FederatedProductWithClusters
                if self.use_cluster_conditional and self.scenario == "vertical":
                    logging.info("\n[Cluster-Conditional Vertical] Gap 4 Integration")
                    logging.info(
                        "  Using FederatedProductWithClusters (Seng Assumption 2)"
                    )
                    logging.info("  Factorization: p(X) = Σₗ q(L=l) Πᵢ p(Xᵢ|L=l)")

                    fed_spn = FederatedProductWithClusters(
                        local_cluster_models=client_local_mixtures,
                        feature_maps=feature_maps,
                        num_features=self.d_features,
                        num_clusters=K_local,
                        device=self.device,
                    )
                    logging.info(
                        f"  ✓ Built cluster-conditional product: {K_local} clusters, {self.K_clients} clients"
                    )
                    self.vertical_feature_map = feature_maps

                else:
                    # UNIFIED MODE: Mixture-of-Products for both Vertical and Hybrid
                    # Vertical is a special case of Hybrid with no overlapping features
                    # P(X1, ..., Xn) = Σ_l q(L=l) × Π_groups p(group | L=l)
                    logging.info(
                        f"[{self.scenario.title()} Mode] Building mixture-of-products (Seng et al. 2025 Algorithm 1)"
                    )
                    if self.scenario == "vertical":
                        logging.info(
                            "  Captures cross-client dependencies via latent cluster variable"
                        )
                    else:
                        logging.info(
                            "  Handles overlapping features via horizontal mixtures within products"
                        )

                    from causallearn.utils.FedPC import (
                        build_feature_indicator_matrix,
                        group_features_by_client_set,
                        GroupMixture,
                        ProductOverGroupsWithOverlap,
                        GlobalSumOfProducts,
                        sample_cluster_combinations,
                    )

                    # Step 1: Sample cluster combinations
                    # Each combination represents a "dependency pattern" (latent state L=l)
                    combinations, combo_weights = sample_cluster_combinations(
                        K_clients=self.K_clients,
                        K_local=K_local,
                        num_samples=None,  # Enumerate all if K_local^K <= 20, else sample
                        seed=42,
                    )

                    logging.info(
                        f"  Building {len(combinations)} products for cluster combinations"
                    )

                    # Step 2: Build feature indicator matrix
                    M, feature_names = build_feature_indicator_matrix(
                        X_splits=X_splits,
                        scenario=self.scenario,
                        d_features=self.d_features,
                    )

                    # Step 3: Group features by client overlap patterns
                    # For vertical: returns {(0,): [f1,f2], (1,): [f3,f4], ...} (no overlaps)
                    # For hybrid: returns {(0,): [...], (0,1): [...], ...} (with overlaps)
                    feature_subspaces = group_features_by_client_set(M, feature_names)

                    logging.info(
                        f"  Feature subspaces: {len(feature_subspaces)} groups"
                    )
                    for client_set, features in feature_subspaces.items():
                        if len(client_set) == 1:
                            logging.info(
                                f"    Client {client_set[0]}: features {features}"
                            )
                        else:
                            logging.info(
                                f"    Clients {client_set} (overlap): features {features}"
                            )

                    # Step 4: Build products for each cluster combination
                    products = []

                    for combo_idx, cluster_config in enumerate(combinations):
                        logging.info(
                            f"  Product {combo_idx + 1}/{len(combinations)}: "
                            f"cluster config={cluster_config}"
                        )

                        # For this cluster configuration, build feature groups
                        group_mixtures = []
                        feature_groups = []

                        for client_set, features in feature_subspaces.items():
                            # Collect SPNs from selected clusters for this feature group
                            num_clients_in_group = len(client_set)
                            cluster_spns_for_group = []

                            for k in client_set:
                                # Get which cluster to use for client k in this configuration
                                cluster_idx = cluster_config[k]

                                # Extract the cluster SPN from LocalClusterMixture
                                cluster_spn = client_local_mixtures[k].cluster_spns[
                                    cluster_idx
                                ]

                                cluster_spns_for_group.append(cluster_spn)

                            if len(cluster_spns_for_group) == 0:
                                logging.warning(
                                    f"    No SPNs for features {features} in combo {combo_idx}, skipping"
                                )
                                continue

                            # Equal weight for each client in the group
                            weight_value = 1.0 / num_clients_in_group
                            group_weights = [weight_value] * num_clients_in_group

                            # Create GroupMixture for this feature subspace
                            # For vertical: single SPN with weight 1.0, extract features (no NaN masking)
                            # For hybrid: multiple SPNs mixed, also extract features (no NaN masking)
                            # IMPORTANT: In hybrid mode, SPNs are trained on LOCAL feature subspaces,
                            # not full d-dimensional space, so we extract features, not mask with NaN
                            group_mix = GroupMixture(
                                client_spns=cluster_spns_for_group,
                                weights=group_weights,
                                feature_indices=features,
                                device=self.device,
                                full_d=self.d_features,  # Total features (for context stripping)
                                use_nan_masking=False,  # Always extract features, never NaN mask
                            )
                            group_mixtures.append(group_mix)
                            feature_groups.append(features)

                        if len(group_mixtures) == 0:
                            raise RuntimeError(
                                f"Combo {combo_idx}: No feature groups created!"
                            )

                        # Create product for this cluster combination
                        # ProductOverGroupsWithOverlap handles both scenarios:
                        # - Vertical: no overlaps, behaves like ProductOverGroups
                        # - Hybrid: with overlaps, handles feature intersection
                        product = ProductOverGroupsWithOverlap(
                            group_mixtures=group_mixtures,
                            feature_groups=feature_groups,
                            device=self.device,
                            allow_overlap=(self.scenario == "hybrid"),
                        )
                        products.append(product)

                        logging.info(
                            f"    ✓ Product {combo_idx + 1}: {len(group_mixtures)} feature groups"
                        )

                    # Step 5: Build mixture over all products
                    # This is the global model: P(X) = Σ_l q(L=l) × p(X | L=l)
                    fed_spn = GlobalSumOfProducts(
                        products=products,
                        weights=combo_weights,
                        device=self.device,
                    )

                    logging.info(
                        f"  ✓ {self.scenario.title()} mixture-of-products built: "
                        f"{len(products)} products, {len(feature_groups)} groups per product"
                    )

                    # Store feature_map for vertical evaluation
                    if self.scenario == "vertical":
                        self.vertical_feature_map = feature_maps

            # Skip the else clause - fed_spn should be set by now from one of the branches above

            # Wrap in FedCDH_SPN_Wrapper for consistency
            self.fed_spn_model = FedCDH_SPN_Wrapper(
                fed_spn, u_index=self.d_features, routing=False
            )

            # Derive covariance tensor from SPN for FICP orientation
            # This replaces the need for parallel KCI summary statistics
            if getattr(self.args, "use_spn_covariances", True):
                logging.info("\n" + "=" * 60)
                logging.info("Deriving Covariances from SPN for Orientation")
                logging.info("=" * 60)
                self.covariance_tensor = self.derive_covariances_from_spn(
                    fed_spn,
                    n_samples=getattr(self.args, "cov_n_samples", 5000),
                    n_fourier_features=getattr(self.args, "cov_n_features", 10),
                    seed=getattr(self.args, "seed", 42),
                )
                logging.info("✓ Covariances derived from SPN successfully\n")
            else:
                self.covariance_tensor = None
                logging.info(
                    "Note: Using KCI-based orientation (use_spn_covariances=False)\n"
                )

            # Store local SPNs for post-hoc quality evaluation AND orientation
            # In new architecture: client_local_mixtures[k] = LocalClusterMixture
            # For vertical mode: store full LocalClusterMixture for ownership-aware orientation
            # For evaluation: extract first cluster SPN
            self.local_cluster_mixtures = client_local_mixtures  # Store full mixtures

            self.local_spns = []
            for k in range(self.K_clients):
                if len(client_local_mixtures[k].cluster_spns) > 0:
                    # Use first local cluster SPN from this client
                    self.local_spns.append(client_local_mixtures[k].cluster_spns[0])
                else:
                    logging.warning(f"Client {k} has no cluster SPNs!")

            logging.info(
                f"Stored {len(self.local_spns)} local SPNs for quality evaluation"
            )

            train_time = time.time() - start_train

        # ============================================================
        # Create Experiment Directory & Logging (ALWAYS)
        # ============================================================
        import os
        from datetime import datetime

        # Create unique output directory for this run (ALWAYS, even if skipping eval)
        if hasattr(self.args, "spn_eval_dir") and self.args.spn_eval_dir:
            output_dir = self.args.spn_eval_dir
        else:
            # Auto-generate: eval/{timestamp}_{scenario}_{K}clients_{d}vars_{n}samples/
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{timestamp}_{self.scenario}_{self.K_clients}clients_{self.d_features}vars_{total_samples}samples"

            # Get project root (3 levels up from FedCDH.py)
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
                )
            )
            output_dir = os.path.join(project_root, "eval", run_id)

        os.makedirs(output_dir, exist_ok=True)
        self.spn_eval_dir = output_dir  # Store for access after fit()

        # Setup file logging for this run
        log_file = os.path.join(output_dir, "run.log")
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Get root logger and add file handler
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        self._log_file_handler = file_handler  # Store to remove later

        # Log run metadata
        logging.info("=" * 60)
        logging.info("FedCDH Run Configuration")
        logging.info("=" * 60)
        logging.info(f"Scenario: {self.scenario}")
        logging.info(f"Clients (K): {self.K_clients}")
        logging.info(f"Features (d): {self.d_features}")
        logging.info(f"Total samples: {total_samples}")
        logging.info(f"CI method: {self.ci_method}")
        logging.info(f"Alpha: {getattr(self.args, 'alpha', 0.05)}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Model type: {self.model_type}")
        logging.info(f"SPN epochs: {getattr(self.args, 'epochs', 'N/A')}")
        logging.info(f"Output directory: {output_dir}")
        logging.info("=" * 60 + "\n")

        # ============================================================
        # SPN Quality Evaluation (MMD, KS tests, UMAP visualization)
        # ============================================================
        # Skip expensive evaluation if skip_spn_eval flag is set (for faster validation tests)
        skip_eval = getattr(self.args, "skip_spn_eval", False)
        if self.ci_method == "spn" and not skip_eval:
            from causallearn.utils.spn_evaluation import (
                evaluate_spn_quality,
                log_spn_quality,
                create_umap_visualization,
            )

            logging.info("\n" + "=" * 60)
            logging.info("SPN Quality Evaluation")
            logging.info("=" * 60)

            # Evaluate local SPNs
            local_eval_results = []  # Collect results for dashboard
            if self.local_spns and len(self.local_spns) > 0:
                logging.info(f"Evaluating {len(self.local_spns)} local SPNs...")

                for k, local_spn in enumerate(self.local_spns):
                    # Prepare client's data with context
                    if self.scenario == "horizontal":
                        samples_per_client = total_samples // self.K_clients
                        X_client = X_global[
                            k * samples_per_client : (k + 1) * samples_per_client, :
                        ]
                        c_client = c_indx[
                            k * samples_per_client : (k + 1) * samples_per_client, :
                        ]
                    elif self.scenario == "vertical":
                        # Vertical: Use the ACTUAL training data for accurate evaluation
                        # Bug fix: Previously reconstructed data from X_global, causing normalization mismatch
                        if hasattr(self, "X_splits_train") and k < len(
                            self.X_splits_train
                        ):
                            X_client_aug = self.X_splits_train[k]
                            # Extract feature indices for logging only
                            feature_indices_no_context = self._extract_feature_indices(
                                k, include_context=False
                            )
                            logging.info(
                                f"  Client {k}: Evaluating on training data (features {feature_indices_no_context}, "
                                f"shape={X_client_aug.shape})"
                            )
                            # Skip the context handling below
                            c_client = None
                        else:
                            # Fallback to old method if training data not available
                            feature_indices_no_context = self._extract_feature_indices(
                                k, include_context=False
                            )

                            if not feature_indices_no_context:
                                logging.warning(
                                    f"  Client {k}: No features found, skipping"
                                )
                                continue

                            # Extract features for this client
                            X_client = X_global[:, feature_indices_no_context]
                            c_client = c_indx  # All samples, context doesn't change

                            logging.info(
                                f"  Client {k}: Evaluating on features {feature_indices_no_context}"
                            )
                    else:  # hybrid
                        samples_per_client = total_samples // self.K_clients
                        X_client = X_global[
                            k * samples_per_client : (k + 1) * samples_per_client, :
                        ]
                        c_client = c_indx[
                            k * samples_per_client : (k + 1) * samples_per_client, :
                        ]

                    # For vertical mode, only client 0 has context column during training
                    # Skip this if we already have X_client_aug from training data
                    if self.scenario == "vertical" and c_client is None:
                        # Already set X_client_aug from training data, skip
                        pass
                    elif self.scenario == "vertical" and k > 0:
                        X_client_aug = X_client  # No context for clients other than 0
                    elif self.scenario == "hybrid":
                        # BUGFIX: Hybrid mode local cluster SPNs don't use context column
                        # They are trained on raw features only (no context)
                        X_client_aug = X_client  # No context column
                    else:
                        X_client_aug = np.concatenate([X_client, c_client], axis=1)

                    # Create descriptive name
                    if self.scenario == "vertical":
                        feature_indices_display = self._extract_feature_indices(
                            k, include_context=False
                        )
                        spn_name = (
                            f"Local SPN Client {k} (Features {feature_indices_display})"
                        )
                    else:
                        spn_name = f"Local SPN Client {k}"

                    # Determine if X_client_aug has context column
                    # Vertical: Only client 0 has context (if using context at all)
                    # Hybrid: No context in local SPNs
                    # Horizontal: All clients have context
                    if self.scenario == "vertical":
                        # Client 0 has context, others don't
                        local_has_context = k == 0
                    elif self.scenario == "hybrid":
                        # Hybrid local SPNs trained without context
                        local_has_context = False
                    else:  # horizontal
                        # Horizontal local SPNs have context
                        local_has_context = True

                    # Evaluate quality
                    result = evaluate_spn_quality(
                        local_spn,
                        X_client_aug,
                        n_samples=min(150, len(X_client_aug)),
                        device=self.device,
                        compute_mmd=True,
                        compute_ks=True,
                        name=spn_name,
                        has_context_column=local_has_context,
                    )

                    log_spn_quality(result)

                    # Store result for dashboard
                    local_eval_results.append(result)

                    # Independence structure evaluation (if ground truth available)
                    if true_DAG_bin is not None:
                        from causallearn.utils.spn_evaluation import (
                            evaluate_spn_independence_structure,
                            log_independence_structure_results,
                        )

                        # Adaptive num_permutations for evaluation
                        eval_perms = min(200, max(50, self.d_features * 10))

                        # For vertical mode, subset true_DAG_bin to client's features
                        if self.scenario == "vertical":
                            feature_indices_no_context = self._extract_feature_indices(
                                k, include_context=False
                            )
                            if len(feature_indices_no_context) > 0:
                                # Extract submatrix for this client's features
                                true_DAG_subset = true_DAG_bin[
                                    np.ix_(
                                        feature_indices_no_context,
                                        feature_indices_no_context,
                                    )
                                ]
                            else:
                                true_DAG_subset = true_DAG_bin
                        else:
                            true_DAG_subset = true_DAG_bin

                        indep_result = evaluate_spn_independence_structure(
                            spn_model=local_spn,
                            X_data=X_client_aug,
                            true_DAG_bin=true_DAG_subset,
                            alpha=0.05,
                            max_order=1,
                            n_conditional_tests=30,
                            num_permutations=eval_perms,
                            device=self.device,
                            name=spn_name,  # Use the same descriptive name
                        )
                        log_independence_structure_results(indep_result)

                        # Merge independence results into quality result
                        local_eval_results[-1].update(indep_result)

                    # UMAP visualization for multivariate data (requires >= 2 features)
                    # For vertical mode, X_client_aug includes context for client 0, so use shape[1]-1 for client 0
                    # For other modes, X_client is defined and used directly
                    if self.scenario == "vertical" and hasattr(self, "X_splits_train"):
                        # Use training data shape minus context column (if present)
                        n_features_no_context = X_client_aug.shape[1] - (
                            1 if k == 0 else 0
                        )
                    else:
                        n_features_no_context = (
                            X_client.shape[1]
                            if "X_client" in locals()
                            else X_client_aug.shape[1] - 1
                        )

                    if n_features_no_context >= 2:
                        with torch.no_grad():
                            samples = (
                                local_spn.sample(min(200, len(X_client_aug)))
                                .cpu()
                                .numpy()
                            )
                        # Remove context column
                        if self.scenario == "vertical" and k == 0:
                            X_features = X_client_aug[
                                :, :-1
                            ]  # Remove context from client 0
                        elif self.scenario == "vertical":
                            X_features = X_client_aug  # Client 1+ has no context
                        else:
                            X_features = (
                                X_client
                                if "X_client" in locals()
                                else X_client_aug[:, :-1]
                            )

                        samples_features = (
                            samples[:, :-1]
                            if samples.shape[1] > X_features.shape[1]
                            else samples
                        )

                        if X_features.shape[1] == samples_features.shape[1]:
                            save_path = os.path.join(
                                output_dir, f"umap_local_client_{k}.png"
                            )
                            # Create descriptive title
                            if self.scenario == "vertical":
                                feature_indices_display = self._extract_feature_indices(
                                    k, include_context=False
                                )
                                umap_title = f"Local SPN (Client {k}, Features {feature_indices_display})"
                            else:
                                umap_title = f"Local SPN (Client {k})"

                            create_umap_visualization(
                                X_features,
                                samples_features,
                                save_path=save_path,
                                title=umap_title,
                            )

            # Evaluate global SPN
            logging.info("Evaluating global federated SPN...")
            # Use stored training data for evaluation (ensures consistent normalization)
            # IMPORTANT: Hybrid mode uses X_global (no context), Horizontal/Vertical use X_aug_global
            if self.scenario == "hybrid":
                # Hybrid models trained without context column
                X_eval = X_global
                has_context = False  # No context column in data or samples
            else:
                # Horizontal and Vertical modes use context column
                # Note: In vertical mode, client 0 SPN is trained with context column U
                X_eval = (
                    self.X_aug_global_train
                    if hasattr(self, "X_aug_global_train")
                    else X_aug_global
                )
                has_context = True  # Context column present
            global_result = evaluate_spn_quality(
                self.fed_spn_model,
                X_eval,
                n_samples=min(300, total_samples),
                device=self.device,
                compute_mmd=True,
                compute_ks=True,
                name="Global Federated SPN",
                has_context_column=has_context,  # Tell evaluation about context
            )

            log_spn_quality(global_result)

            # Independence structure evaluation for global SPN (if ground truth available)
            if true_DAG_bin is not None:
                from causallearn.utils.spn_evaluation import (
                    evaluate_spn_independence_structure,
                    log_independence_structure_results,
                )

                # Adaptive num_permutations for evaluation
                eval_perms = min(200, max(50, self.d_features * 10))

                global_indep_result = evaluate_spn_independence_structure(
                    spn_model=self.fed_spn_model,
                    X_data=X_eval,
                    true_DAG_bin=true_DAG_bin,
                    alpha=0.05,
                    max_order=1,
                    n_conditional_tests=50,  # More tests for global
                    num_permutations=eval_perms,
                    device=self.device,
                    name="Global Federated SPN",
                )
                log_independence_structure_results(global_indep_result)

                # Merge independence results into global result
                global_result.update(global_indep_result)

            # UMAP for global SPN (requires >= 2 features)
            if self.d_features >= 2:
                with torch.no_grad():
                    samples = (
                        self.fed_spn_model.sample(min(300, total_samples)).cpu().numpy()
                    )
                # Remove context column
                X_features = X_global
                samples_features = (
                    samples[:, :-1] if samples.shape[1] > X_global.shape[1] else samples
                )

                if X_features.shape[1] == samples_features.shape[1]:
                    save_path = os.path.join(output_dir, "umap_global_spn.png")
                    create_umap_visualization(
                        X_features,
                        samples_features,
                        save_path=save_path,
                        title="Global Federated SPN",
                    )

            # Generate comprehensive dashboard and HTML report
            logging.info("=" * 60)
            logging.info("Generating SPN Quality Dashboard...")
            logging.info("=" * 60)

            try:
                from causallearn.utils.spn_dashboard import (
                    compute_summary_statistics,
                    create_summary_dashboard,
                    generate_html_report,
                )

                # Compute summary statistics across local SPNs
                summary_stats = compute_summary_statistics(local_eval_results)

                # Configuration info for dashboard
                config_info = {
                    "scenario": self.scenario,
                    "K": self.K_clients,
                    "d": self.d_features,
                    "n": total_samples,
                    "model_type": self.model_type,
                    "ci_method": self.ci_method,
                    "alpha": alpha,
                    "device": str(self.device),
                    "epochs": train_epochs,
                }

                # Create dashboard plot
                dashboard_path = os.path.join(output_dir, "dashboard.png")
                create_summary_dashboard(
                    local_eval_results,
                    global_result,
                    summary_stats,
                    dashboard_path,
                    config_info,
                )

                # Collect UMAP paths
                umap_paths = {}
                for i in range(self.K_clients):
                    path = os.path.join(output_dir, f"umap_local_client_{i}.png")
                    if os.path.exists(path):
                        umap_paths[f"Local Client {i}"] = path
                global_umap = os.path.join(output_dir, "umap_global_spn.png")
                if os.path.exists(global_umap):
                    umap_paths["Global SPN"] = global_umap

                # Generate HTML report
                html_path = generate_html_report(
                    local_eval_results,
                    global_result,
                    summary_stats,
                    config_info,
                    output_dir,
                    dashboard_path,
                    umap_paths,
                )

                logging.info("=" * 60)
                logging.info("Dashboard and report generation complete!")
                logging.info(f"  Dashboard: {dashboard_path}")
                logging.info(f"  HTML Report: {html_path}")
                logging.info("=" * 60)

            except Exception as e:
                logging.warning(f"Dashboard generation failed: {e}")

            logging.info(f"\nSPN evaluation plots saved to: {output_dir}/")
            logging.info(f"Run log saved to: {log_file}")
            logging.info("=" * 60 + "\n")

            # Remove file handler to avoid accumulation across runs
            if hasattr(self, "_log_file_handler"):
                file_handler.flush()
                file_handler.close()
                root_logger.removeHandler(file_handler)
                delattr(self, "_log_file_handler")

        start_cd = time.time()
        from causallearn.utils.cit import CIT, SPN_CIT

        # Permutation test configuration
        # CRITICAL FIX: Parametric test (num_permutations=0) uses df=1 which is incorrect
        # for continuous SPNs. This causes false negatives on standardized data (e.g., Asia).
        # Default: num_permutations=50 (non-parametric, statistically correct)
        # Override via args.num_permutations if specified
        num_permutations = getattr(self.args, "num_permutations", 50)
        if num_permutations > 0:
            logging.info(
                f"Using permutation test with num_permutations={num_permutations} (statistically correct)"
            )
        else:
            logging.warning(
                "Using parametric chi-square test (num_permutations=0) - may produce incorrect p-values on standardized data"
            )

        # Causal discovery using global CI test
        if self.ci_method == "spn":
            cit_obj = SPN_CIT(
                X_aug_global,
                global_model=self.fed_spn_model,
                num_permutations=num_permutations,
            )
        elif self.ci_method == "kci":
            cit_obj = CIT(X_aug_global, "kci")
        else:
            cit_obj = CIT(X_aug_global, "fisherz")

        cit_counter = QueryCounterCIT(cit_obj)

        # Adaptive depth limit based on dataset characteristics
        # Prevents excessive conditioning set sizes that hurt SPN accuracy
        # Rule of thumb: max_depth ≈ log(n) / 2, capped at 3-4
        n_samples = X_aug_global.shape[0]
        default_depth_limit = min(4, max(2, int(np.log(n_samples) / 2)))
        depth_limit = getattr(self.args, "depth_limit", default_depth_limit)
        logging.info(
            f"Using depth_limit={depth_limit} for skeleton discovery (n={n_samples}, d={self.d_features})"
        )

        # Prepare kwargs for cdnod
        cdnod_kwargs = {
            "num_permutations": 0,
            "orientation_type": getattr(self.args, "ablation_orientation", "mi_hybrid"),
            "depth_limit": depth_limit,
            "verbose": True,  # Enable verbose logging to see skeleton initialization
        }

        # FIX #3: Pass initial skeleton from structure voting if available
        if (
            hasattr(self, "initial_skeleton_from_voting")
            and self.initial_skeleton_from_voting is not None
        ):
            cdnod_kwargs["initial_skeleton"] = self.initial_skeleton_from_voting
            n_edges = self.initial_skeleton_from_voting.sum() // 2
            logging.info(
                f"[Structure Voting] Passing initial skeleton to PC algorithm: {n_edges} edges from consensus graph"
            )

        # Pass covariance tensor if available (derived from SPN)
        if hasattr(self, "covariance_tensor") and self.covariance_tensor is not None:
            cdnod_kwargs["covariance_tensor"] = self.covariance_tensor
            logging.info("Using SPN-derived covariances for FICP orientation")

        # BUGFIX: Pass feature_maps and local_cluster_mixtures for vertical mode orientation
        # Vertical mode partitions features, not samples, so orientation needs feature ownership
        # Ownership-aware orientation uses local SPNs (within-client) and global product SPN (cross-client)
        if self.scenario == "vertical" and hasattr(self, "vertical_feature_map"):
            cdnod_kwargs["feature_maps"] = self.vertical_feature_map
            logging.info(
                f"Vertical mode: Passing feature_maps to cdnod for ownership-aware orientation (K={len(self.vertical_feature_map)} clients)"
            )

            # Pass local cluster mixtures if available (stored during training)
            if (
                hasattr(self, "local_cluster_mixtures")
                and self.local_cluster_mixtures is not None
            ):
                cdnod_kwargs[
                    "local_spns"
                ] = self.local_cluster_mixtures  # Pass LocalClusterMixture objects
                logging.info(
                    f"Vertical mode: Passing {len(self.local_cluster_mixtures)} local cluster mixtures for within-client edge orientation"
                )

        # FIX #2: Exclude augmented variable from skeleton in horizontal mode
        # The augmented variable (client ID) is used for conditioning but should not appear in the causal graph
        exclude_augmented = self.scenario == "horizontal"

        cg = cdnod(
            X_global,
            c_indx,
            self.K_clients,
            alpha=alpha,
            indep_test=cit_counter,
            stable=True,
            uc_rule=2,
            uc_priority=-1,
            fed_spn_model=self.fed_spn_model,
            exclude_augmented_var=exclude_augmented,
            **cdnod_kwargs,
        )
        cd_time = time.time() - start_cd
        num_queries = cit_counter.query_count
        est_graph = cg.G.graph[0 : self.d_features, 0 : self.d_features]
        est_cpdag = get_cpdag_from_cdnod(est_graph)
        est_dag = get_dag_from_pdag(est_cpdag)

        if self.ci_method == "spn":
            comm_cost = (
                estimate_fedcdh_comm_cost(
                    self.d_features,
                    self.K_clients,
                    num_clusters,
                    model_size_params=5000,
                )
                + clustering_cost
            )
        elif self.ci_method == "kci":
            comm_cost = estimate_kci_comm_cost(
                total_samples, self.K_clients, num_queries
            )
        res_skel = count_skeleton_accuracy(true_DAG_bin, est_cpdag)
        res_dir = count_dag_accuracy(true_DAG_bin, est_dag)
        result = {
            **res_skel,
            **res_dir,
            "time_train": train_time,
            "time_cd": cd_time,
            "comm_cost": comm_cost,
        }

        # Optional: Return discovered graphs for visualization
        if getattr(self.args, "return_graphs", False):
            result["est_dag"] = est_dag
            result["est_cpdag"] = est_cpdag
            result["true_dag"] = true_DAG_bin

        safe_result = {
            k: (
                v  # Keep arrays (graphs) as-is
                if isinstance(v, np.ndarray)
                else float(v)
                if v is not None
                and isinstance(v, (int, float))
                and not (isinstance(v, float) and np.isnan(v))
                else 0.0
                if isinstance(v, (int, float))
                else v  # Keep other non-numeric values
            )
            for k, v in result.items()
        }
        return safe_result
