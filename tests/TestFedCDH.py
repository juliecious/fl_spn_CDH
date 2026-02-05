import sys
import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import logging

sys.path.append("")

from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.FedPC import (
    GlobalFedSPN,
    LocalSPNWrapper,
)
from causallearn.utils.data_utils import (
    my_simulate_general_hetero,
    my_simulate_linear_gaussian,
    count_dag_accuracy,
    count_skeleton_accuracy,
    get_cpdag_from_cdnod,
    get_dag_from_pdag,
    set_random_seed,
    simulate_dag,
)

np.set_printoptions(suppress=True, precision=3)


class FedCDH_SPN_Wrapper(nn.Module):
    """
    Wraps GlobalFedSPN to handle 'Context/Client Index' (U) explicitly.
    Supports two modes:
    1. routing=True (Horizontal): P(X, U=k) = P(U=k) * P(X | Component_k)
    2. routing=False (Vertical/Hybrid): P(X, U) is modeled jointly as P(Features)
    """

    def __init__(self, global_spn: GlobalFedSPN, u_index: int, routing: bool = True):
        super().__init__()
        self.spn = global_spn
        self.u_index = u_index
        self.device = global_spn.device
        self.routing = routing

    def log_prob(self, x):
        if not self.routing:
            return self.spn.log_prob(x)

        if self.u_index == -1 or self.u_index == x.shape[1] - 1:
            x_feat = x[:, :-1]
            u_col = x[:, -1]
        else:
            x_feat = torch.cat([x[:, : self.u_index], x[:, self.u_index + 1 :]], dim=1)
            u_col = x[:, self.u_index]

        u_is_observed = not torch.isnan(u_col[0]).item()

        if u_is_observed:
            client_indices = u_col.long()
            unique_clients = torch.unique(client_indices)
            final_ll = torch.zeros(x.shape[0], 1, device=self.device)

            for k in unique_clients:
                k_idx = k.item()
                mask = client_indices == k
                x_sub = x_feat[mask]
                ll_sub = self.spn.log_prob_conditional_u(x_sub, k_idx)
                weight = self.spn.weights[k_idx]
                ll_total = ll_sub + torch.log(weight + 1e-9)
                final_ll[mask] = ll_total
            return final_ll
        else:
            return self.spn.log_prob(x_feat)

    def sample(self, n):
        if not self.routing:
            return self.spn.sample(n).view(n, -1)

        u_indices = torch.multinomial(self.spn.weights, n, replacement=True)
        unique_u, counts = torch.unique(u_indices, return_counts=True)

        test_x = self.spn.components[0].sample(1)
        d = test_x.shape[1]
        samples = torch.zeros(n, d + 1, device=self.device)

        start_idx = 0
        for u_val, count in zip(unique_u, counts):
            k = u_val.item()
            c = count.item()
            x_samples = self.spn.components[k].sample(c).view(c, -1)

            end_idx = start_idx + c
            if self.u_index == -1 or self.u_index == d:
                samples[start_idx:end_idx, :d] = x_samples
                samples[start_idx:end_idx, d] = u_val.float()
            else:
                samples[start_idx:end_idx, : self.u_index] = x_samples[
                    :, : self.u_index
                ]
                samples[start_idx:end_idx, self.u_index] = u_val.float()
                samples[start_idx:end_idx, self.u_index + 1 :] = x_samples[
                    :, self.u_index :
                ]
            start_idx = end_idx

        return samples.view(n, -1)


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

        # Determine global dimensionality and samples
        # Note: In Vertical, X_splits[k] has shape (N, d_k). In Horizontal, (N_k, D).
        if scenario == "vertical":
            # Assume all clients have same N rows
            N = X_splits[0].shape[0]
            # Max index in feature_maps to find D
            D = 0
            for k in feature_maps:
                D = max(D, max(feature_maps[k]) + 1)
        else:
            # Horizontal/Hybrid
            N = sum(len(x) for x in X_splits)
            D = X_splits[0].shape[1]

        # Initialize Centroids (Server-side)
        # Random uniform initialization
        # For simulation, we cheat slightly to get bounds, but in real FL server would init blindly or ask for bounds (negligible cost)
        if scenario == "vertical":
            # Need bounds from all clients to init global centroids
            # Cost: 2 * D floats (min/max) -> negligible
            g_min, g_max = np.full(D, np.inf), np.full(D, -np.inf)
            for k in range(K_clients):
                cols = feature_maps[k]
                l_min, l_max = X_splits[k].min(axis=0), X_splits[k].max(axis=0)
                g_min[cols] = np.minimum(g_min[cols], l_min)
                g_max[cols] = np.maximum(g_max[cols], l_max)
        else:
            x0 = X_splits[0]
            g_min, g_max = x0.min(axis=0), x0.max(axis=0)  # Simplified

        centroids = np.random.uniform(g_min, g_max, (self.n_clusters, D))

        # Iteration Loop
        for it in range(self.max_iter):
            prev_centroids = centroids.copy()

            if scenario == "vertical":
                # --- Vertical FL K-Means ---
                # 1. Server sends partial centroids to clients
                # Cost: K_clients * (K_clusters * D_k) * 4
                for k in range(K_clients):
                    d_k = len(feature_maps[k])
                    self.comm_cost += self.n_clusters * d_k * 4

                # 2. Clients compute partial squared distances
                # ||x - c||^2 = \sum_k ||x_k - c_k||^2
                partial_dists = np.zeros((N, self.n_clusters))

                for k in range(K_clients):
                    cols = feature_maps[k]
                    # (N, d_k) - (n_clust, d_k)
                    # Expand to (N, n_clust, d_k)
                    dist_k = np.sum(
                        (X_splits[k][:, None, :] - centroids[None, :, cols]) ** 2,
                        axis=2,
                    )
                    partial_dists += dist_k

                # 3. Clients send partial distances to Server
                # Cost: K_clients * (N * K_clusters) * 4  <-- Expensive!
                self.comm_cost += K_clients * N * self.n_clusters * 4

                # 4. Server assigns labels
                self.labels_ = np.argmin(partial_dists, axis=1)
                self.inertia_ = np.min(partial_dists, axis=1).sum()

                # 5. Server sends labels back to clients
                # Cost: K_clients * N * 4 (int)
                self.comm_cost += K_clients * N * 4

                # 6. Clients compute partial sums
                # 7. Server aggregates
                new_centroids = np.zeros_like(centroids)
                global_counts = np.zeros(self.n_clusters)

                # Counts are same for all clients (based on labels), can be computed by server or one client
                # Let's say server computes counts.
                for c in range(self.n_clusters):
                    global_counts[c] = (self.labels_ == c).sum()

                for k in range(K_clients):
                    cols = feature_maps[k]
                    # Compute sum for each cluster
                    for c in range(self.n_clusters):
                        mask = self.labels_ == c
                        if global_counts[c] > 0:
                            partial_sum = X_splits[k][mask].sum(axis=0)
                            new_centroids[c, cols] = partial_sum / global_counts[c]

                    # Cost: K_clusters * D_k * 4 (sending means/sums)
                    self.comm_cost += self.n_clusters * len(cols) * 4

            else:
                # --- Horizontal/Hybrid FL K-Means ---
                global_sums = np.zeros((self.n_clusters, D))
                global_counts = np.zeros(self.n_clusters)
                self.inertia_ = 0

                all_labels = []

                # 1. Server sends Centroids to Clients
                # Cost: K_clients * (K_clusters * D) * 4
                self.comm_cost += K_clients * self.n_clusters * D * 4

                for k in range(K_clients):
                    if len(X_splits[k]) == 0:
                        all_labels.append(np.array([]))
                        continue

                    # 2. Client computes distances & labels locally
                    dists = (
                        np.linalg.norm(
                            X_splits[k][:, None, :] - centroids[None, :, :], axis=2
                        )
                        ** 2
                    )
                    lbs = np.argmin(dists, axis=1)
                    all_labels.append(lbs)

                    self.inertia_ += np.min(dists, axis=1).sum()

                    # 3. Client computes sums/counts
                    for c in range(self.n_clusters):
                        mask = lbs == c
                        count = mask.sum()
                        if count > 0:
                            global_sums[c] += X_splits[k][mask].sum(axis=0)
                            global_counts[c] += count

                self.labels_ = (
                    np.concatenate(all_labels) if all_labels else np.array([])
                )

                # 4. Clients send Sums/Counts to Server
                # Cost: K_clients * (K_clusters * (D + 1)) * 4
                self.comm_cost += K_clients * self.n_clusters * (D + 1) * 4

                # 5. Server updates centroids
                new_centroids = np.zeros_like(centroids)
                for c in range(self.n_clusters):
                    if global_counts[c] > 0:
                        new_centroids[c] = global_sums[c] / global_counts[c]
                    else:
                        new_centroids[c] = centroids[c]

            # Convergence Check
            if np.linalg.norm(new_centroids - prev_centroids) < self.tol:
                break
            centroids = new_centroids

        return self


def test_fedCDH(i, args):
    n_samples_per_client = args.n
    K_clients = args.K
    d_features = args.d
    s0 = args.d
    model_type = args.model_type
    ci_method = args.ci_method
    scenario = args.scenario
    device = "cpu"

    use_structure = getattr(args, "ablation_structure", True)
    use_em = getattr(args, "ablation_em", True)
    orientation_type = getattr(args, "ablation_orientation", "hybrid")

    set_random_seed(i)

    # 1. Data Generation
    from tests.utils.benchmark_loaders import (
        load_standard_graph,
        simulate_heterogeneous_data,
    )

    if model_type == "sachs_real":
        # --- Real-World Interventional Benchmark ---
        from tests.utils.sachs_loader import load_sachs_federated

        X_splits, true_DAG_bin, c_indx = load_sachs_federated(
            K_clients, n_samples_limit=n_samples_per_client * K_clients
        )

        # Reconstruct Global for KCI (Centralized Baseline)
        X_global = np.concatenate(X_splits, axis=0)
        d_features = X_global.shape[1]

        # Prepare aggregated data for clustering/hybrid logic
        X_aug_global = np.concatenate([X_global, c_indx], axis=1)
        d_aug_total = X_aug_global.shape[1]

        # Augment X_splits with U for SPN training
        # (Since downstream logic expects X_splits to align with feature_maps including U)
        X_splits_aug = []
        for k, x_k in enumerate(X_splits):
            c_k = c_indx[c_indx == k].reshape(-1, 1)
            # Safety: Ensure lengths match (they should from loader)
            if len(x_k) != len(c_k):
                # Fallback if c_indx wasn't perfectly aligned by loader order
                c_k = np.full((len(x_k), 1), k)
            X_splits_aug.append(np.concatenate([x_k, c_k], axis=1))
        X_splits = X_splits_aug

    elif model_type in ["sachs", "asia", "alarm"]:
        # Standard Graph Structure + Synthetic Heterogeneity
        true_DAG_bin = load_standard_graph(model_type)
        d_features = true_DAG_bin.shape[0]  # Override d with actual graph size

        # Simulate Heterogeneous Data from this Structure
        # We assume 'general' non-linear mechanisms by default for benchmarks
        X_global, c_indx = simulate_heterogeneous_data(
            true_DAG_bin, K_clients, n_samples_per_client, mode="general"
        )

        # Ensure c_indx is int
        c_indx = c_indx.astype(int)

        # Split for Federated
        X_splits = np.array_split(X_global, K_clients)

    else:
        # Random Synthetic Graph Path
        true_DAG_bin = simulate_dag(d_features, s0, "ER")
        total_samples = n_samples_per_client * K_clients

        if model_type == "linear":
            X_global, c_indx = my_simulate_linear_gaussian(
                true_DAG_bin, K_clients, total_samples, "gauss"
            )
        else:
            X_global, c_indx = my_simulate_general_hetero(
                true_DAG_bin, K_clients, total_samples, "gauss"
            )

        # Normalization and Indexing (aligned with standard path)
        c_indx = np.repeat(np.arange(K_clients), n_samples_per_client).reshape(-1, 1)
        X_global = (X_global - X_global.mean(0)) / (X_global.std(0) + 1e-6)

    total_samples = X_global.shape[0]

    train_time = 0
    fed_spn_model = None
    routing_mode = True
    clustering_cost = 0

    if ci_method == "voting_pc":
        # --- Baseline: Voting-based Federated PC ---
        start_cd = time.time()
        local_graphs = []
        for k in range(K_clients):
            # 1. Prepare Local Data
            local_X = X_splits[k]
            # Handle potential augmented columns (remove context U if present for standard PC)
            if local_X.shape[1] > d_features:
                local_data = local_X[:, :d_features]
            else:
                local_data = local_X

            # 2. Run Local PC (Standard FisherZ)
            # We use cdnod wrapper but with K=1 and local data
            # Dummy context for API compatibility
            local_c = np.zeros((len(local_data), 1))

            try:
                cg_local = cdnod(
                    local_data,
                    local_c,
                    1,
                    alpha=0.01,
                    indep_test="fisherz",
                    stable=True,
                    uc_rule=2,
                    uc_priority=-1,
                )
                local_graphs.append(cg_local.G.graph[0:d_features, 0:d_features])
            except Exception:
                # Fallback empty graph if local PC fails
                local_graphs.append(np.zeros((d_features, d_features)))

        # 3. Majority Vote Aggregation
        global_skeleton = np.zeros((d_features, d_features))
        for g in local_graphs:
            # Symmetrize and binarize
            skel = ((g + g.T) != 0).astype(int)
            global_skeleton += skel

        # Threshold > K/2
        consensus_skeleton = (global_skeleton > (K_clients / 2)).astype(int)

        # 4. Construct Result
        est_cpdag = np.zeros((d_features, d_features))
        rows, cols = np.where(consensus_skeleton == 1)
        for r, c in zip(rows, cols):
            est_cpdag[r, c] = -1  # Undirected edge

        est_dag = np.zeros_like(est_cpdag)  # No orientation in naive voting

        cd_time = time.time() - start_cd
        train_time = 0.0

        # Metrics
        res_skel = count_skeleton_accuracy(true_DAG_bin, est_cpdag)
        res_dir = count_dag_accuracy(true_DAG_bin, est_dag)
        # 4 bytes per entry for float adjacency
        comm_cost = (K_clients * d_features * d_features * 4) / 1024.0

        result = {
            **res_skel,
            **res_dir,
            "time_train": train_time,
            "time_cd": cd_time,
            "comm_cost": comm_cost,
        }

        safe_result = {}
        for k, v in result.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                safe_result[k] = 0.0
            else:
                safe_result[k] = float(v)
        logging.info(f"   Result: Skel F1={safe_result.get('f1_skeleton', 0):.2f}")
        return safe_result

    elif ci_method == "spn":
        # ... (Existing SPN logic) ...
        # (This block remains unchanged, just showing context)
        pass

    elif ci_method == "voting_pc":
        # --- Baseline: Voting-based Federated PC ---
        # Clients run PC locally, Server aggregates by Majority Vote
        start_cd = time.time()

        local_graphs = []
        # Pre-compute correlation matrices for speed if needed, but PC is fast

        for k in range(K_clients):
            # Local PC
            # We use standard fisherz for local independence test
            # Note: We pass c_indx slice just to satisfy cdnod signature,
            # but standard PC ignores it or we can use standard PC.
            # Using cdnod locally to be fair (handling heterogeneity if local data has it)

            # Slice local data
            local_X = X_splits[k]
            # Create dummy local context (all 0s since we are inside one client)
            # Or use actual if available. For 'sachs_real', X_splits might be augmented already?
            # Let's check dimensions.
            if local_X.shape[1] > d_features:
                local_data = local_X[:, :d_features]
                local_c = local_X[:, d_features].reshape(-1, 1)
            else:
                local_data = local_X
                local_c = np.zeros((len(local_X), 1))

            # Run Local Discovery
            # We use a simple PC-stable for robustness
            cg_local = cdnod(
                local_data,
                local_c,
                1,  # K=1 locally
                alpha=0.01,
                indep_test="fisherz",
                stable=True,
                uc_rule=2,
                uc_priority=-1,
            )
            local_graphs.append(cg_local.G.graph[0:d_features, 0:d_features])

        # Aggregation: Majority Vote
        # Edges are {1, -1}. We check for presence (non-zero).
        # We aggregate skeletons first.
        global_skeleton = np.zeros((d_features, d_features))

        for g in local_graphs:
            # Binarize skeleton: 1 if edge exists (1 or -1), 0 otherwise
            skel = (g != 0).astype(int)
            # Symmetrize to be safe
            skel = ((skel + skel.T) > 0).astype(int)
            global_skeleton += skel

        # Threshold: > K/2 votes
        consensus_skeleton = (global_skeleton > (K_clients / 2)).astype(int)

        # Construct Result Graph (CPDAG format for evaluation)
        # We only output the skeleton for this baseline as orientation voting is complex
        # and usually performed by local orientation rules which might conflict.
        # We return an undirected graph where edges exist.
        est_cpdag = np.zeros((d_features, d_features))
        # Set symmetric -1 for edges
        rows, cols = np.where(consensus_skeleton == 1)
        for r, c in zip(rows, cols):
            est_cpdag[r, c] = -1

        cd_time = time.time() - start_cd

        # Evaluation
        # Voting PC produces a Skeleton. Orientation is not aggregated here (F1 Dir will be low).
        # This is a fair "Structure-Only" baseline.
        est_dag = np.zeros_like(est_cpdag)  # Dummy DAG

        res_skel = count_skeleton_accuracy(true_DAG_bin, est_cpdag)
        res_dir = count_dag_accuracy(true_DAG_bin, est_dag)  # Will be 0

        # Communication Cost: K clients * Adj Matrix size (d*d bits/bytes)
        # 4 bytes per entry for float adjacency
        comm_cost = (K_clients * d_features * d_features * 4) / 1024.0

        train_time = 0.0

        # Skip the main SPN block logic
        fed_spn_model = None

    if ci_method == "spn":
        start_train = time.time()
        from causallearn.utils.FedPC import FederatedStructureLearner, FederatedProduct

        # Unified Federated Data Partitioning (Seng 2025)
        # We treat all scenarios as learning a joint distribution P(X, U) via clustering.
        routing_mode = False
        X_aug_global = np.concatenate([X_global, c_indx], axis=1)
        d_aug_total = X_aug_global.shape[1]

        # 1. Define Feature Maps per scenario
        # If we loaded real federated data, X_splits is already set.
        if model_type == "sachs_real":
            feature_maps = {k: list(range(d_aug_total)) for k in range(K_clients)}
            # X_splits is already augmented and split from above
        elif scenario == "horizontal":
            feature_maps = {k: list(range(d_aug_total)) for k in range(K_clients)}
            X_splits = np.array_split(X_aug_global, K_clients)
        elif scenario == "vertical":
            cols_per_client = np.array_split(range(d_features), K_clients)
            feature_maps = {}
            X_splits = []
            for k in range(K_clients):
                f_indices = cols_per_client[k].tolist()
                if k == 0:
                    f_indices.append(d_features)  # Client 0 gets U
                feature_maps[k] = f_indices
                X_splits.append(X_aug_global[:, f_indices])
        else:  # hybrid
            feature_maps = {k: list(range(d_aug_total)) for k in range(K_clients)}
            X_splits = np.array_split(X_aug_global, K_clients)

        # 2. Federated Clustering (Simulated)
        # Find best K using BIC
        best_h = 2
        min_bic = float("inf")
        best_model = None

        # Heuristic for K=2..4 clusters
        for h_candidate in range(2, 4):
            fed_km = SimulatedFederatedKMeans(
                n_clusters=h_candidate, max_iter=10, seed=42
            )
            fed_km.fit(X_splits, feature_maps, scenario)

            # BIC = Inertia + k * log(N) * D
            bic = fed_km.inertia_ + h_candidate * np.log(total_samples) * d_aug_total

            if bic < min_bic:
                min_bic = bic
                best_h = h_candidate
                best_model = fed_km

        num_clusters = best_h
        labels = best_model.labels_
        clustering_cost = best_model.comm_cost / 1024.0  # KB

        # Split labels to clients to simulate local knowledge of cluster assignment
        # Use actual split lengths to handle uneven datasets (like Sachs interventional)
        labels_splits = []
        _curr = 0
        for _xk in X_splits:
            labels_splits.append(labels[_curr : _curr + len(_xk)])
            _curr += len(_xk)

        weights = np.bincount(labels, minlength=num_clusters) / len(labels)

        # 3. Train Local Components
        clients_clusters = [[] for _ in range(num_clusters)]
        clients_counts = [[] for _ in range(num_clusters)]

        for h in range(num_clusters):
            cluster_mask_global = labels == h
            if cluster_mask_global.sum() < 5:
                continue

            for k in range(K_clients):
                if scenario == "vertical":
                    local_data_h = X_splits[k][cluster_mask_global]
                else:
                    local_data_h = X_splits[k][labels_splits[k] == h]

                if len(local_data_h) > 2:
                    local_d = local_data_h.shape[1]
                    leaf = LocalSPNWrapper(
                        num_features=local_d,
                        device=device,
                        num_sums=5,
                        num_leaves=5,
                        depth=max(1, int(np.floor(np.log2(local_d)))),
                        num_repetitions=5,
                        seed=i * 100 + h * 10 + k,
                    )
                    leaf.train_local(local_data_h, epochs=1, lr=0.01)
                    clients_clusters[h].append(leaf)
                    clients_counts[h].append(len(local_data_h))

        # 4. Aggregation (The Remedy)
        global_components = []
        final_weights = []

        for h in range(num_clusters):
            if not clients_clusters[h]:
                continue

            # Check for feature overlap
            is_disjoint = True
            if len(feature_maps) > 1:
                s0 = set(feature_maps[0])
                s1 = set(feature_maps[1])
                if not s0.isdisjoint(s1):
                    is_disjoint = False

            if is_disjoint and len(clients_clusters[h]) == K_clients:
                # Vertical -> Product (Independent across clients)
                comp = FederatedProduct(
                    clients_clusters[h], feature_map=feature_maps, device=device
                )
                global_components.append(comp)
                final_weights.append(weights[h])
            elif not is_disjoint:
                # Horizontal/Hybrid -> Mixture (Ensemble over clients)
                inner_ws = np.array(clients_counts[h])
                inner_ws = inner_ws / inner_ws.sum()
                comp = GlobalFedSPN(
                    clients_clusters[h],
                    weights=inner_ws,
                    strategy="mixture",
                    device=device,
                )
                global_components.append(comp)
                final_weights.append(weights[h])

        if not final_weights:
            # Fallback if no components were successfully built
            global_spn = GlobalFedSPN([], device=device)
        else:
            final_weights = np.array(final_weights)
            final_weights /= final_weights.sum()
            global_spn = GlobalFedSPN(
                global_components, weights=final_weights, device=device
            )

        if use_em and len(global_components) > 0:
            global_spn.train_weights_em(
                torch.tensor(X_aug_global, dtype=torch.float32).to(device)
            )

        fed_spn_model = FedCDH_SPN_Wrapper(
            global_spn, u_index=d_features, routing=routing_mode
        )
        train_time = time.time() - start_train

    # 3. Causal Discovery
    start_cd = time.time()
    cg = cdnod(
        X_global,
        c_indx,
        K_clients,
        alpha=0.01,
        indep_test=ci_method,
        stable=True,
        uc_rule=2,
        uc_priority=-1,
        fed_spn_model=fed_spn_model,
        num_permutations=20,
        orientation_type=orientation_type,
    )
    cd_time = time.time() - start_cd

    est_graph = cg.G.graph[0:d_features, 0:d_features]
    est_cpdag = get_cpdag_from_cdnod(est_graph)
    est_dag = get_dag_from_pdag(est_cpdag)

    res_skel = count_skeleton_accuracy(true_DAG_bin, est_cpdag)
    res_dir = count_dag_accuracy(true_DAG_bin, est_dag)

    comm_cost = (
        fed_spn_model.spn.get_total_communication_cost() / 1024.0
        if fed_spn_model
        else 0.0
    )
    # Add clustering cost
    comm_cost += clustering_cost

    result = {
        **res_skel,
        **res_dir,
        "time_train": train_time,
        "time_cd": cd_time,
        "comm_cost": comm_cost,
    }

    # Final Sanitization to prevent logging/formatting crashes
    safe_result = {}
    for k, v in result.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            safe_result[k] = 0.0
        else:
            safe_result[k] = float(v)

    logging.info(
        f"   Result: Skel F1={safe_result.get('f1_skeleton', 0):.2f} | Dir F1={safe_result.get('f1', 0):.2f}"
    )
    return safe_result


if __name__ == "__main__":
    # Standard entry point if run directly
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="horizontal", type=str)
    parser.add_argument("--ci_method", default="spn", type=str)
    parser.add_argument("--n", default=200, type=int)
    parser.add_argument("--K", default=2, type=int)
    parser.add_argument("--d", default=5, type=int)
    parser.add_argument("--model_type", default="general", type=str)
    args = parser.parse_args()
    test_fedCDH(0, args)
