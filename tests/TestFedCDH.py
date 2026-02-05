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

    c_indx = np.repeat(np.arange(K_clients), n_samples_per_client).reshape(-1, 1)
    X_global = (X_global - X_global.mean(0)) / (X_global.std(0) + 1e-6)

    train_time = 0
    fed_spn_model = None
    routing_mode = True

    if ci_method == "spn":
        start_train = time.time()
        from causallearn.utils.FedPC import FederatedStructureLearner, FederatedProduct

        # Unified Federated Data Partitioning (Seng 2025)
        # We treat all scenarios as learning a joint distribution P(X, U) via clustering.
        routing_mode = False
        X_aug_global = np.concatenate([X_global, c_indx], axis=1)
        d_aug_total = X_aug_global.shape[1]

        # 1. Define Feature Maps per scenario
        if scenario == "horizontal":
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

        # 2. Global Clustering (Proxy for Federated KMeans)
        from sklearn.cluster import KMeans

        best_h = 2
        min_bic = float("inf")
        for h_candidate in range(2, 7):
            km = KMeans(n_clusters=h_candidate, n_init=3, random_state=42).fit(
                X_aug_global
            )
            bic = km.inertia_ + h_candidate * np.log(total_samples) * d_aug_total
            if bic < min_bic:
                min_bic = bic
                best_h = h_candidate

        num_clusters = best_h
        kmeans = KMeans(n_clusters=num_clusters, n_init=5, random_state=42).fit(
            X_aug_global
        )
        labels = kmeans.labels_
        labels_splits = np.array_split(labels, K_clients)
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
                        num_sums=10,
                        num_leaves=10,
                        depth=max(1, int(np.floor(np.log2(local_d)))),
                        num_repetitions=5,
                        seed=i * 100 + h * 10 + k,
                    )
                    leaf.train_local(local_data_h, epochs=2, lr=0.01)
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
        num_permutations=50,
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
