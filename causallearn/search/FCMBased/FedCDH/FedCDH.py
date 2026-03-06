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
            N = X_splits[0].shape[0]
            D = 0
            for k in feature_maps:
                D = max(D, max(feature_maps[k]) + 1)
        else:
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


class QueryCounterCIT:
    def __init__(self, cit_instance):
        self.cit = cit_instance
        self.query_count = 0
        self.method = getattr(cit_instance, "method", "unknown")

    def __call__(self, *args, **kwargs):
        self.query_count += 1
        if self.method in ["spn", "kci"]:
            return self.cit(*args, **kwargs)
        else:
            return self.cit(*args[:3])

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.cit, name)


class FedCDH:
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        # Reverting to CPU: MPS fails on 5D tensor reductions in simple-einet
        self.device = torch.device("cpu")

        logging.info(f"FedCDH Initialized on device: {self.device}")
        self.K_clients = args.K
        self.d_features = args.d
        self.scenario = args.scenario
        self.model_type = args.model_type
        self.ci_method = args.ci_method
        self.n_samples_per_client = args.n
        self.fed_spn_model = None

    def fit(self, X_splits, c_indx, true_DAG_bin):
        # Get training epochs and alpha from args
        if hasattr(self.args, "epochs"):
            train_epochs = self.args.epochs
        else:
            train_epochs = 50 if self.device.type in ["cuda", "gpu"] else 10
        alpha = self.args.alpha if hasattr(self.args, "alpha") else 0.05

        # Reconstruct Global for KCI/Oracle baselines
        if isinstance(X_splits, list):
            X_global = np.concatenate(X_splits, axis=0)
        else:
            X_global = X_splits
        total_samples = X_global.shape[0]
        X_aug_global = np.concatenate([X_global, c_indx], axis=1)
        d_aug_total = X_aug_global.shape[1]
        train_time = 0
        cd_time = 0
        comm_cost = 0.0
        clustering_cost = 0.0
        num_queries = 0

        if self.ci_method == "spn":
            start_train = time.time()
            if self.scenario == "horizontal":
                feature_maps = {
                    k: list(range(d_aug_total)) for k in range(self.K_clients)
                }
                # Keep existing splits (e.g. from sachs_real) but ensure they include the context column U
                new_splits = []
                _curr = 0
                for xk in X_splits:
                    new_splits.append(X_aug_global[_curr : _curr + len(xk)])
                    _curr += len(xk)
                X_splits = new_splits
            elif self.scenario == "vertical":
                cols_per_client = np.array_split(range(self.d_features), self.K_clients)
                feature_maps = {}
                X_splits_train = []
                for k in range(self.K_clients):
                    f_indices = cols_per_client[k].tolist()
                    if k == 0:
                        f_indices.append(self.d_features)
                    feature_maps[k] = f_indices
                    X_splits_train.append(X_aug_global[:, f_indices])
                X_splits = X_splits_train
            else:  # hybrid
                feature_maps = {
                    k: list(range(d_aug_total)) for k in range(self.K_clients)
                }
                # Always use X_aug_global to ensure context U is included in training data
                X_splits = np.array_split(X_aug_global, self.K_clients)

            best_h = 2
            min_bic = float("inf")
            best_model = None
            for h_candidate in range(2, 6):
                fed_km = SimulatedFederatedKMeans(
                    n_clusters=h_candidate, max_iter=10, seed=42
                )
                fed_km.fit(X_splits, feature_maps, self.scenario)
                bic = (
                    fed_km.inertia_ + h_candidate * np.log(total_samples) * d_aug_total
                )
                if bic < min_bic:
                    min_bic = bic
                    best_h = h_candidate
                    best_model = fed_km
            num_clusters = best_h
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
            # Get training epochs and alpha from args
            if hasattr(self.args, "epochs"):
                train_epochs = self.args.epochs
            else:
                train_epochs = 50 if self.device.type in ["cuda", "gpu"] else 10
            alpha = self.args.alpha if hasattr(self.args, "alpha") else 0.05
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
            for h in range(num_clusters):
                if not clients_clusters[h]:
                    continue
                is_disjoint = True
                if len(feature_maps) > 1:
                    if not set(feature_maps[0]).isdisjoint(set(feature_maps[1])):
                        is_disjoint = False
                if is_disjoint and len(clients_clusters[h]) == self.K_clients:
                    comp = FederatedProduct(
                        clients_clusters[h],
                        feature_map=feature_maps,
                        device=self.device,
                    )
                    global_components.append(comp)
                    final_weights.append(weights[h])
                elif not is_disjoint:
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
            train_time = time.time() - start_train

        start_cd = time.time()
        from causallearn.utils.cit import CIT, SPN_CIT

        if self.ci_method == "voting_pc":
            local_graphs = []
            for k in range(self.K_clients):
                local_X = X_splits[k]
                local_data = (
                    local_X[:, : self.d_features]
                    if local_X.shape[1] > self.d_features
                    else local_X
                )
                local_c = (
                    local_X[:, self.d_features].reshape(-1, 1)
                    if local_X.shape[1] > self.d_features
                    else np.zeros((len(local_X), 1))
                )
                try:
                    cg_local = cdnod(
                        local_data,
                        local_c,
                        1,
                        alpha=alpha,
                        indep_test="fisherz",
                        stable=True,
                        uc_rule=2,
                        uc_priority=-1,
                    )
                    local_graphs.append(
                        cg_local.G.graph[0 : self.d_features, 0 : self.d_features]
                    )
                except:
                    local_graphs.append(np.zeros((self.d_features, self.d_features)))
            global_skeleton = np.zeros((self.d_features, self.d_features))
            for g in local_graphs:
                skel = ((g + g.T) != 0).astype(int)
                global_skeleton += skel
            consensus = (global_skeleton > (self.K_clients / 2)).astype(int)
            est_cpdag = np.zeros((self.d_features, self.d_features))
            rows, cols = np.where(consensus == 1)
            for r, c in zip(rows, cols):
                est_cpdag[r, c] = -1
            est_dag = np.zeros_like(est_cpdag)
            cd_time = time.time() - start_cd
            comm_cost = (self.K_clients * self.d_features**2 * 4) / 1024.0
        else:
            if self.ci_method == "spn":
                cit_obj = SPN_CIT(
                    X_aug_global, global_model=self.fed_spn_model, num_permutations=0
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
