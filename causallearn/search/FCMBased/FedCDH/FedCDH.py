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
    UnivariateSPNWrapper,
    FederatedProduct,
    FederatedStructureLearner,
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
                    # Use X_aug_global to ensure context U is included
                    X_splits = np.array_split(X_aug_global, self.K_clients)

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

            # BIC-based cluster selection (optional: skip if K specified)
            # If args.skip_bic=True, use K_clients as number of clusters
            # Otherwise, run BIC selection over [2, ..., 5] to find optimal K
            if getattr(self.args, "skip_bic", False):
                # Skip BIC: use user-specified K directly
                num_clusters = self.K_clients
                logging.info(
                    f"BIC selection skipped. Using K={num_clusters} clusters directly."
                )
                fed_km = SimulatedFederatedKMeans(
                    n_clusters=num_clusters, max_iter=10, seed=42
                )
                fed_km.fit(X_splits, feature_maps, self.scenario)
                best_model = fed_km
            else:
                # Run BIC selection
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
                num_clusters = best_h
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
            # Extract SPN architecture parameters
            num_sums = getattr(self.args, "num_sums", 5)
            num_leaves = getattr(self.args, "num_leaves", 5)
            num_repetitions = getattr(self.args, "num_repetitions", 5)

            for h in range(num_clusters):
                cluster_mask_global = labels == h
                if cluster_mask_global.sum() < 5:
                    continue
                for k in range(self.K_clients):
                    local_data_h = (
                        X_splits[k][cluster_mask_global]
                        if self.scenario == "vertical"
                        else X_splits[k][labels_splits[k] == h]
                    )
                    if len(local_data_h) > 2:
                        local_d = local_data_h.shape[1]
                        if local_d == 1:
                            leaf = UnivariateSPNWrapper(
                                device=self.device,
                                num_sums=num_sums,
                                num_leaves=num_leaves,
                                seed=h * 10 + k,
                            )
                        else:
                            leaf = LocalSPNWrapper(
                                num_features=local_d,
                                device=self.device,
                                num_sums=num_sums,
                                num_leaves=num_leaves,
                                depth=max(1, int(np.floor(np.log2(local_d)))),
                                num_repetitions=num_repetitions,
                                seed=h * 10 + k,
                            )
                        leaf.train_local(local_data_h, epochs=train_epochs, lr=0.01)
                        clients_clusters[h].append(leaf)
                        clients_counts[h].append(len(local_data_h))
            global_components = []
            final_weights = []

            # Simplified Hybrid: Product-then-Mixture
            # Check if hybrid scenario with feature groups
            is_hybrid_mode = (
                self.scenario == "hybrid"
                and hasattr(self, "feature_groups")
                and self.feature_groups is not None
            )

            if is_hybrid_mode:
                logging.info(
                    f"Using simplified hybrid aggregation with {len(self.feature_groups)} feature groups"
                )

                # For each cluster, build product-then-mixture
                for h in range(num_clusters):
                    if not clients_clusters[h]:
                        continue

                    # Each cluster has K clients * num_feature_groups SPNs
                    # Group them by client
                    num_groups = len(self.feature_groups)

                    # Rebuild clients_clusters[h] with feature group structure
                    # Current: clients_clusters[h] is flat list of SPNs
                    # We need to re-train with feature groups

                    client_products = []
                    client_counts_updated = []

                    # Get cluster data for re-training with feature groups
                    cluster_mask_global = labels == h
                    if cluster_mask_global.sum() < 5:
                        continue

                    for k in range(self.K_clients):
                        # Get local data for this cluster and client
                        local_data_h = X_splits[k][labels_splits[k] == h]

                        if len(local_data_h) <= 2:
                            continue

                        # Train one SPN per feature group for this client
                        group_spns = []
                        for g, feature_group in enumerate(self.feature_groups):
                            # Extract data for this feature group
                            local_data_group = local_data_h[:, feature_group]
                            local_d_group = local_data_group.shape[1]

                            if local_d_group == 1:
                                leaf = UnivariateSPNWrapper(
                                    device=self.device,
                                    num_sums=num_sums,
                                    num_leaves=num_leaves,
                                    seed=h * 100 + k * 10 + g,
                                )
                            else:
                                leaf = LocalSPNWrapper(
                                    num_features=local_d_group,
                                    device=self.device,
                                    num_sums=num_sums,
                                    num_leaves=num_leaves,
                                    depth=max(1, int(np.floor(np.log2(local_d_group)))),
                                    num_repetitions=num_repetitions,
                                    seed=h * 100 + k * 10 + g,
                                )

                            leaf.train_local(
                                local_data_group, epochs=train_epochs, lr=0.01
                            )
                            group_spns.append(leaf)

                        # Product over feature groups for this client
                        if len(group_spns) > 1:
                            # Convert feature_groups list to dict format for FederatedProduct
                            # FederatedProduct expects {0: [features_g0], 1: [features_g1], ...}
                            feature_map_dict = {
                                g: self.feature_groups[g]
                                for g in range(len(self.feature_groups))
                            }

                            client_product = FederatedProduct(
                                group_spns,
                                feature_map=feature_map_dict,
                                device=self.device,
                            )
                            client_products.append(client_product)
                            client_counts_updated.append(len(local_data_h))
                        elif len(group_spns) == 1:
                            # Only one group, use directly
                            client_products.append(group_spns[0])
                            client_counts_updated.append(len(local_data_h))

                    # Mixture over clients (their products)
                    if client_products:
                        inner_ws = np.array(client_counts_updated)
                        inner_ws = inner_ws / inner_ws.sum()
                        comp = GlobalFedSPN(
                            client_products,
                            weights=inner_ws,
                            strategy="mixture",
                            device=self.device,
                        )
                        global_components.append(comp)
                        final_weights.append(weights[h])

            else:
                # Original vertical/horizontal logic
                for h in range(num_clusters):
                    if not clients_clusters[h]:
                        continue

                    # Check if features are disjoint (only for vertical scenario)
                    is_disjoint = (
                        feature_maps is not None
                        and len(feature_maps) > 1
                        and set(feature_maps[0]).isdisjoint(set(feature_maps[1]))
                    )

                    if is_disjoint and len(clients_clusters[h]) == self.K_clients:
                        # Vertical scenario: Use FederatedProduct for disjoint features
                        comp = FederatedProduct(
                            clients_clusters[h],
                            feature_map=feature_maps,
                            device=self.device,
                        )
                        global_components.append(comp)
                        final_weights.append(weights[h])
                        # Store feature_map for vertical evaluation
                        self.vertical_feature_map = feature_maps
                    else:
                        inner_ws = np.array(clients_counts[h])
                        inner_ws = inner_ws / inner_ws.sum()
                        comp = GlobalFedSPN(
                            clients_clusters[h],
                            weights=inner_ws,
                            strategy="mixture",
                            device=self.device,
                        )
                        global_components.append(comp)
                        final_weights.append(weights[h])
            if not final_weights:
                global_spn = GlobalFedSPN([], device=self.device)
            else:
                final_weights = np.array(final_weights)
                final_weights /= final_weights.sum()
                global_spn = GlobalFedSPN(
                    global_components, weights=final_weights, device=self.device
                )
            if len(global_components) > 0:
                global_spn.train_weights_em(
                    torch.tensor(X_aug_global, dtype=torch.float32).to(self.device)
                )
            self.fed_spn_model = FedCDH_SPN_Wrapper(
                global_spn, u_index=self.d_features, routing=False
            )

            # Store local SPNs for post-hoc quality evaluation
            # Extract one representative SPN per client from the clustering structure
            # clients_clusters[h][k] = SPN trained on cluster h at client k
            # For evaluation, we need K SPNs (one per client) to assess local training quality
            self.local_spns = []
            for k in range(self.K_clients):
                # Find the first SPN trained on client k's data across all clusters
                for h in range(num_clusters):
                    if clients_clusters[h] and len(clients_clusters[h]) > k:
                        self.local_spns.append(clients_clusters[h][k])
                        break
                else:
                    # Fallback: if no SPN found for this client, use first available
                    # (Happens if client has too few samples after clustering)
                    for h in range(num_clusters):
                        if clients_clusters[h]:
                            self.local_spns.append(clients_clusters[h][0])
                            break

            train_time = time.time() - start_train

        # ============================================================
        # SPN Quality Evaluation (MMD, KS tests, UMAP visualization)
        # ============================================================
        if self.ci_method == "spn":
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
                        # Vertical: all samples, subset of features
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
                    if self.scenario == "vertical" and k > 0:
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
                        n_samples=min(150, len(X_client)),
                        device=self.device,
                        compute_mmd=True,
                        compute_ks=True,
                        name=spn_name,
                    )

                    log_spn_quality(result)

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

                    # UMAP visualization for multivariate data
                    if X_client.shape[1] > 2:
                        with torch.no_grad():
                            samples = (
                                local_spn.sample(min(200, len(X_client))).cpu().numpy()
                            )
                        # Remove context column
                        X_features = X_client
                        samples_features = (
                            samples[:, :-1]
                            if samples.shape[1] > X_client.shape[1]
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
            global_result = evaluate_spn_quality(
                self.fed_spn_model,
                X_aug_global,
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
                    X_data=X_aug_global,
                    true_DAG_bin=true_DAG_bin,
                    alpha=0.05,
                    max_order=1,
                    n_conditional_tests=50,  # More tests for global
                    num_permutations=eval_perms,
                    device=self.device,
                    name="Global Federated SPN",
                )
                log_independence_structure_results(global_indep_result)

            # UMAP for global SPN
            if self.d_features > 2:
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

            logging.info(f"SPN evaluation plots saved to: {output_dir}/")
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

        # Adaptive num_permutations based on dimensionality
        # For d≥10, need more permutations for reliable p-value calibration
        # Formula: min(200, max(50, d × 10))
        num_permutations = min(200, max(50, self.d_features * 10))
        if num_permutations > 50:
            logging.info(
                f"Using adaptive num_permutations={num_permutations} for d={self.d_features}"
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
