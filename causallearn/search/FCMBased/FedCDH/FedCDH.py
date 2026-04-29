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
    FederatedStructureLearner,
    GroupMixture,
    ProductOverGroups,
    ProductOverGroupsWithOverlap,
    build_feature_indicator_matrix,
    group_features_by_client_set,
    compute_adaptive_hyperparameters,
)
from causallearn.utils.data_utils import (
    count_dag_accuracy,
    count_skeleton_accuracy,
    get_cpdag_from_cdnod,
    get_dag_from_pdag,
    set_random_seed,
)
from causallearn.utils.cost_analysis import (
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
        else:
            # Horizontal/Hybrid: all clients see all features
            N = sum(len(x) for x in X_splits)
            D = X_splits[0].shape[1]

        if scenario == "vertical":
            g_min, g_max = np.full(D, np.inf), np.full(D, -np.inf)
            for k in range(K_clients):
                cols = feature_maps[k]
                l_min, l_max = X_splits[k].min(axis=0), X_splits[k].max(axis=0)
                g_min[cols] = np.minimum(g_min[cols], l_min)
                g_max[cols] = np.maximum(g_max[cols], l_max)
        else:
            x0 = X_splits[0]
            if len(x0) > 0:
                g_min, g_max = x0.min(axis=0), x0.max(axis=0)
            else:
                g_min, g_max = np.zeros(D), np.ones(D)

        centroids = np.random.uniform(g_min, g_max, (self.n_clusters, D))

        for it in range(self.max_iter):
            prev_centroids = centroids.copy()
            if scenario == "vertical":
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
                global_sums = np.zeros((self.n_clusters, D))
                global_counts = np.zeros(self.n_clusters)
                self.inertia_ = 0
                all_labels = []
                self.comm_cost += K_clients * self.n_clusters * D * 4
                for k in range(K_clients):
                    if len(X_splits[k]) == 0:
                        all_labels.append(np.array([]))
                        continue
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
    def __init__(self, args: Dict[str, Any]):
        self.args = args

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
            else:
                # Horizontal/Hybrid: concatenate samples (axis=0)
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

            # Feature maps only needed for vertical scenario (disjoint features)
            # For horizontal/hybrid, all clients see all features
            if self.scenario == "vertical":
                # Vertical: Split features across clients
                cols_per_client = np.array_split(range(self.d_features), self.K_clients)
                feature_maps = {}
                X_splits_train = []
                for k in range(self.K_clients):
                    f_indices = cols_per_client[k].tolist()
                    if k == 0:
                        f_indices.append(self.d_features)  # Context column U
                    feature_maps[k] = f_indices
                    X_splits_train.append(X_aug_global[:, f_indices])
                X_splits = X_splits_train
                # Store for evaluation (fixes vertical mode LL evaluation bug)
                self.X_splits_train = X_splits_train
            else:
                # Horizontal/Hybrid: All clients see all features (no feature map needed)
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
            if self.scenario in ["horizontal", "hybrid"]:
                assert total_samples_split == X_aug_global.shape[0], (
                    f"{self.scenario.capitalize()} scenario: Total samples across clients "
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

            # V2 LOCAL CLUSTERING: Seng et al. (2025) Algorithm 1
            # KEY CHANGE: Client-first loop (not cluster-first) with LOCAL k-means per client
            # Rationale: Prevents data fragmentation, preserves sample sufficiency
            # Reference: experiments/SPN_STRUCTURE_COMPARISON.md Section 1.1

            from sklearn.cluster import KMeans

            # Determine local cluster count
            K_local = getattr(self.args, "num_local_clusters", 2)
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

                        # Create and train SPN for this local cluster
                        if local_d == 1:
                            spn_kh = UnivariateSPNWrapper(
                                device=self.device,
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

                    if local_d == 1:
                        single_spn = UnivariateSPNWrapper(
                            device=self.device,
                            num_sums=num_sums,
                            num_leaves=num_leaves,
                            seed=k * 10,
                        )
                        single_spn.train_local(
                            client_data, epochs=hyperparams["epochs"], lr=adaptive_lr
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
                # P(X) = Σ_k w_k × P_k(X)
                # where P_k = LocalClusterMixture for client k
                logging.info(
                    "[Horizontal Mode] Building global mixture over client mixtures"
                )

                # Dataset weights (proportional to sample counts)
                dataset_weights = np.array(
                    [len(X_splits[k]) for k in range(self.K_clients)]
                )
                dataset_weights = dataset_weights / dataset_weights.sum()

                logging.info(f"  Dataset weights: {dataset_weights}")

                # Global mixture
                fed_spn = GlobalFedSPN(
                    components=client_local_mixtures,
                    weights=dataset_weights.tolist(),
                    strategy="mixture",
                    device=self.device,
                )

                logging.info("  ✓ Horizontal global mixture built")

            elif self.scenario == "vertical":
                # VERTICAL MODE: Product over disjoint feature groups
                # P(X) = Π_k P_k(X_k)
                # where each client owns disjoint features
                logging.info(
                    "[Vertical Mode] Building product over disjoint feature groups"
                )

                # Build GroupMixtures (one per client's feature subset)
                from causallearn.utils.FedPC import GroupMixture, ProductOverGroups

                group_mixtures = []
                feature_groups = []

                for k in range(self.K_clients):
                    # Each client is a group (disjoint features)
                    group_mix = GroupMixture(
                        client_spns=[client_local_mixtures[k]],
                        weights=[1.0],  # Single client
                        feature_indices=feature_maps[k],
                        device=self.device,
                    )
                    group_mixtures.append(group_mix)
                    feature_groups.append(feature_maps[k])

                # Product over disjoint groups
                fed_spn = ProductOverGroups(
                    group_mixtures=group_mixtures,
                    feature_groups=feature_groups,
                    device=self.device,
                )

                logging.info(f"  Feature groups: {feature_groups}")
                logging.info("  ✓ Vertical product built")

                # Store feature_map for vertical evaluation
                self.vertical_feature_map = feature_maps

            elif self.scenario == "hybrid":
                # HYBRID MODE: Special handling - train SPNs per feature subspace
                # Note: Cannot reuse client_local_mixtures because they were trained on
                # full client features, but hybrid mode needs SPNs per feature GROUP
                logging.info(
                    "[Hybrid Mode] Building product with overlap resolution (Algorithm 1)"
                )
                logging.info(
                    "  Note: Training feature-specific SPNs (not reusing global mixtures)"
                )

                from causallearn.utils.FedPC import (
                    build_feature_indicator_matrix,
                    group_features_by_client_set,
                    GroupMixture,
                    ProductOverGroupsWithOverlap,
                )

                # Step 1: Build indicator matrix
                # Note: Only consider causal features (exclude context column U)
                # Context U will be added separately to all groups after grouping
                M, feature_names = build_feature_indicator_matrix(
                    X_splits=X_splits, scenario="hybrid", d_features=self.d_features
                )

                # Step 2: Group features by client set
                feature_subspaces = group_features_by_client_set(M, feature_names)

                # Step 3: Train SPNs per (client, feature_subspace) pair
                group_mixtures = []
                feature_groups = []

                for client_set, features in feature_subspaces.items():
                    logging.info(
                        f"  Training SPNs for features {features} (clients {client_set})"
                    )

                    trained_spns = []
                    client_counts = []

                    for k in client_set:
                        # Get client k's full data
                        client_data = X_splits[k]

                        # Extract ONLY the features for this subspace
                        # Note: Context U will be added later to avoid double-counting
                        client_data_subspace = client_data[:, features]
                        local_d = client_data_subspace.shape[1]
                        n_samples = len(client_data_subspace)

                        if n_samples < 5:
                            logging.warning(
                                f"    Client {k}: insufficient data ({n_samples} samples)"
                            )
                            continue

                        logging.info(
                            f"    Client {k}: training on {n_samples} samples × {local_d} features"
                        )

                        # Get adaptive hyperparameters for this subspace
                        hyperparams = compute_adaptive_hyperparameters(
                            mode=self.scenario,
                            num_features=local_d,
                            num_samples=n_samples,
                            data_type=self.data_type,
                            base_num_sums=num_sums,
                            base_num_leaves=num_leaves,
                            base_epochs=train_epochs,
                        )

                        # Train SPN on this feature subspace
                        if local_d == 1:
                            spn_subspace = UnivariateSPNWrapper(
                                device=self.device,
                                num_sums=num_sums,
                                num_leaves=num_leaves,
                                seed=k * 1000 + hash(tuple(features)) % 1000,
                            )
                            spn_subspace.train_local(
                                client_data_subspace,
                                epochs=hyperparams["epochs"],
                                lr=adaptive_lr,
                            )
                        else:
                            spn_subspace = LocalSPNWrapper(
                                num_features=local_d,
                                device=self.device,
                                num_sums=hyperparams["num_sums"],
                                num_leaves=hyperparams["num_leaves"],
                                depth=hyperparams["depth"],
                                num_repetitions=num_repetitions,
                                seed=k * 1000 + hash(tuple(features)) % 1000,
                            )
                            spn_subspace.train_local(
                                client_data_subspace,
                                epochs=hyperparams["epochs"],
                                lr=adaptive_lr,
                                l1_weight=1e-4,
                                l2_weight=hyperparams["weight_decay"],
                                dropout=hyperparams["dropout"],
                            )

                        trained_spns.append(spn_subspace)
                        client_counts.append(n_samples)

                    if len(trained_spns) == 0:
                        logging.warning(
                            f"  No SPNs trained for features {features}, skipping"
                        )
                        continue

                    # Compute weights (proportional to sample counts)
                    group_weights = np.array(client_counts)
                    group_weights = group_weights / group_weights.sum()

                    # Create GroupMixture for this feature subspace
                    group_mix = GroupMixture(
                        client_spns=trained_spns,
                        weights=group_weights.tolist(),
                        feature_indices=features,
                        device=self.device,
                    )
                    group_mixtures.append(group_mix)
                    feature_groups.append(features)

                    logging.info(
                        f"    ✓ Feature group {features}: {len(trained_spns)} SPNs, weights={group_weights}"
                    )

                if len(group_mixtures) == 0:
                    raise RuntimeError("Hybrid mode: No feature groups created!")

                # Step 4: Product with overlap resolution
                fed_spn = ProductOverGroupsWithOverlap(
                    group_mixtures=group_mixtures,
                    feature_groups=feature_groups,
                    device=self.device,
                    allow_overlap=True,
                )

                logging.info(
                    f"  ✓ Hybrid product built: {len(group_mixtures)} feature groups"
                )

            else:
                raise ValueError(f"Unknown scenario: {self.scenario}")

            # Wrap in FedCDH_SPN_Wrapper for consistency
            self.fed_spn_model = FedCDH_SPN_Wrapper(
                fed_spn, u_index=self.d_features, routing=False
            )

            # Store local SPNs for post-hoc quality evaluation
            # In new architecture: client_local_mixtures[k] = LocalClusterMixture
            # Each LocalClusterMixture contains cluster_spns[h] = SPNs for local clusters
            # For evaluation, we use the first local cluster SPN from each client
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
            import os
            from datetime import datetime

            logging.info("\n" + "=" * 60)
            logging.info("SPN Quality Evaluation")
            logging.info("=" * 60)

            # Create unique output directory for this run
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

                    # Evaluate quality
                    result = evaluate_spn_quality(
                        local_spn,
                        X_client_aug,
                        n_samples=min(150, len(X_client_aug)),
                        device=self.device,
                        compute_mmd=True,
                        compute_ks=True,
                        name=spn_name,
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
            # IMPORTANT: Hybrid/vertical modes use X_global (no context), horizontal uses X_aug_global
            if self.scenario in ["vertical", "hybrid"]:
                # Vertical/hybrid models trained without context column
                X_eval = X_global
            else:
                # Horizontal mode uses context column for routing
                X_eval = (
                    self.X_aug_global_train
                    if hasattr(self, "X_aug_global_train")
                    else X_aug_global
                )
            global_result = evaluate_spn_quality(
                self.fed_spn_model,
                X_eval,
                n_samples=min(300, total_samples),
                device=self.device,
                compute_mmd=True,
                compute_ks=True,
                name="Global Federated SPN",
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
        # Default: num_permutations=0 (parametric chi-square test, faster)
        # Override via args.num_permutations if specified
        # v2 change: Smoke tests showed parametric = permutation (both F1=0.133)
        # Permutation test adds no value but costs 50x compute per CI test
        num_permutations = getattr(self.args, "num_permutations", 0)
        if num_permutations > 0:
            logging.info(
                f"Using permutation test with num_permutations={num_permutations}"
            )
        else:
            logging.info("Using parametric chi-square test (num_permutations=0)")

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
            num_permutations=0,
            orientation_type=getattr(self.args, "ablation_orientation", "hybrid"),
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
        safe_result = {
            k: (
                float(v)
                if v is not None and not (isinstance(v, float) and np.isnan(v))
                else 0.0
            )
            for k, v in result.items()
        }
        return safe_result
