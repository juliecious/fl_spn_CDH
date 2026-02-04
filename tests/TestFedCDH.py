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
        """
        x: [Batch, D+1] tensor.
        """
        if not self.routing:
            # Joint mode: U is just another feature in the SPN
            return self.spn.log_prob(x)

        # Routing mode (Horizontal)
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
        """
        Sample [X, U] from the hierarchical model.
        """
        if not self.routing:
            # Joint mode: SPN already produces [X, U]
            return self.spn.sample(n)

        # Routing mode (Horizontal)
        u_indices = torch.multinomial(self.spn.weights, n, replacement=True)
        unique_u, counts = torch.unique(u_indices, return_counts=True)

        test_x = self.spn.components[0].sample(1)
        d = test_x.shape[1]
        samples = torch.zeros(n, d + 1, device=self.device)

        start_idx = 0
        for u_val, count in zip(unique_u, counts):
            k = u_val.item()
            c = count.item()
            x_samples = self.spn.components[k].sample(c)
            x_samples = x_samples.view(c, -1)

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

    def ci_test(self, *args, **kwargs):
        pass


def test_fedCDH(i, args):
    n_samples_per_client = args.n
    K_clients = args.K
    d_features = args.d
    s0 = args.d
    model_type = args.model_type
    ci_method = args.ci_method
    scenario = args.scenario
    device = "cpu"  # or "cuda" if available

    # Ablation Flags (Defaults to True/Hybrid if not provided)
    use_structure = getattr(args, "ablation_structure", True)
    use_em = getattr(args, "ablation_em", True)
    orientation_type = getattr(args, "ablation_orientation", "hybrid")

    set_random_seed(i)
    logging.info(
        f"Running Instance {i} | K={K_clients} | Scenario={scenario} | "
        f"Ablations: Struct={use_structure}, EM={use_em}, Orient={orientation_type}"
    )

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

    # Ensure c_indx is correct shape/type
    c_indx = np.repeat(np.arange(K_clients), n_samples_per_client).reshape(-1, 1)

    # Simple global normalization for stability
    X_global = (X_global - X_global.mean(0)) / (X_global.std(0) + 1e-6)

    logging.info(">>> Phase 1: Density Estimation (FedSPN Training)...")
    train_time = 0
    fed_spn_model = None
    routing_mode = True

    if ci_method == "spn":
        start_train = time.time()

        X_splits = []
        feature_maps = {}
        num_clusters = 5

        from causallearn.utils.FedPC import FederatedStructureLearner

        struct_learner = FederatedStructureLearner(num_features=d_features)

        if scenario == "horizontal":
            routing_mode = True
            X_splits = np.array_split(X_global, K_clients)
            for k in range(K_clients):
                feature_maps[k] = list(range(d_features))
                if use_structure:
                    struct_learner.add_local_metadata(X_splits[k])

            causal_order = (
                struct_learner.get_causal_order()
                if use_structure
                else np.arange(d_features)
            )
            local_models = []
            for k in range(K_clients):
                local_data = X_splits[k]
                leaf = LocalSPNWrapper(
                    num_features=local_data.shape[1],
                    device=device,
                    num_sums=20,
                    num_leaves=20,
                    depth=max(1, int(np.floor(np.log2(local_data.shape[1])))),
                    num_repetitions=5,
                    seed=i * 100 + k,
                    variable_order=causal_order,
                )
                leaf.train_local(local_data, epochs=30, lr=0.01)
                local_models.append(leaf)

            global_spn = GlobalFedSPN(
                local_models,
                device=device,
                feature_map=feature_maps,
                strategy="mixture",
            )

        elif scenario == "vertical":
            routing_mode = False  # Joint mode
            # Include C in Global Data for Vertical
            X_aug_global = np.concatenate([X_global, c_indx], axis=1)
            d_aug_total = X_aug_global.shape[1]

            # Split features, give C to the first client
            cols_per_client = np.array_split(range(d_features), K_clients)
            for k in range(K_clients):
                f_indices = cols_per_client[k].tolist()
                if k == 0:
                    f_indices.append(d_features)  # Append C index
                feature_maps[k] = f_indices
                X_splits.append(X_aug_global[:, f_indices])

            from sklearn.cluster import KMeans

            best_h = 2
            min_bic = float("inf")
            for h_candidate in range(2, 9):
                km = KMeans(n_clusters=h_candidate, n_init=5, random_state=42).fit(
                    X_aug_global
                )
                bic = km.inertia_ + h_candidate * np.log(total_samples) * d_aug_total
                if bic < min_bic:
                    min_bic = bic
                    best_h = h_candidate

            num_clusters = best_h
            kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42).fit(
                X_aug_global
            )
            labels = kmeans.labels_
            weights = np.bincount(labels, minlength=num_clusters) / len(labels)

            clients_clusters = [[] for _ in range(num_clusters)]
            for h in range(num_clusters):
                cluster_mask = labels == h
                if cluster_mask.sum() < 10:
                    continue
                for k in range(K_clients):
                    local_data_h = X_splits[k][cluster_mask]
                    local_d = local_data_h.shape[1]
                    local_order = np.arange(local_d)
                    if use_structure:
                        local_struct = FederatedStructureLearner(num_features=local_d)
                        local_struct.add_local_metadata(local_data_h)
                        local_order = local_struct.get_causal_order()

                    leaf = LocalSPNWrapper(
                        num_features=local_d,
                        device=device,
                        num_sums=10,
                        num_leaves=10,
                        depth=max(1, int(np.floor(np.log2(local_d)))),
                        num_repetitions=5,
                        seed=i * 100 + h * 10 + k,
                        variable_order=local_order,
                    )
                    leaf.train_local(local_data_h, epochs=20, lr=0.01)
                    clients_clusters[h].append(leaf)

            from causallearn.utils.FedPC import FederatedProduct

            products = []
            for h in range(num_clusters):
                if len(clients_clusters[h]) == K_clients:
                    prod = FederatedProduct(
                        clients_clusters[h], feature_map=feature_maps, device=device
                    )
                    products.append(prod)

            global_spn = GlobalFedSPN(
                products, weights=weights[: len(products)], device=device
            )
            if use_em:
                global_spn.train_weights_em(
                    torch.tensor(X_aug_global, dtype=torch.float32).to(device)
                )

        elif scenario == "hybrid":
            routing_mode = False  # Joint mode
            X_aug_global = np.concatenate([X_global, c_indx], axis=1)
            d_aug_total = X_aug_global.shape[1]
            X_splits = np.array_split(X_aug_global, K_clients)
            feature_maps = {k: list(range(d_aug_total)) for k in range(K_clients)}

            causal_order = np.arange(d_aug_total)
            if use_structure:
                for k in range(K_clients):
                    struct_learner.add_local_metadata(X_splits[k])
                causal_order = struct_learner.get_causal_order()

            local_models = []
            for k in range(K_clients):
                local_data = X_splits[k]
                leaf = LocalSPNWrapper(
                    num_features=d_aug_total,
                    device=device,
                    num_sums=20,
                    num_leaves=20,
                    depth=max(1, int(np.floor(np.log2(d_aug_total)))),
                    num_repetitions=5,
                    seed=i * 100 + k,
                    variable_order=causal_order,
                )
                leaf.train_local(local_data, epochs=30, lr=0.01)
                local_models.append(leaf)
            global_spn = GlobalFedSPN(
                local_models,
                device=device,
                feature_map=feature_maps,
                strategy="mixture",
            )
            if use_em:
                global_spn.train_weights_em(
                    torch.tensor(X_aug_global, dtype=torch.float32).to(device)
                )

        fed_spn_model = FedCDH_SPN_Wrapper(
            global_spn, u_index=d_features, routing=routing_mode
        )
        train_time = time.time() - start_train
        logging.info(f"   FedSPN Training Complete ({train_time:.2f}s)")

    else:
        indep_test_obj = ci_method

    # 3. Causal Discovery
    logging.info(">>> Phase 2: Causal Discovery (FedCDH)...")
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

    if fed_spn_model is not None and hasattr(
        fed_spn_model.spn, "get_total_communication_cost"
    ):
        comm_cost = fed_spn_model.spn.get_total_communication_cost() / 1024.0
    else:
        comm_cost = 0.0

    result = {
        **res_skel,
        **res_dir,
        "time_train": train_time,
        "time_cd": cd_time,
        "comm_cost": comm_cost,
    }

    skel_f1 = result.get("f1_skeleton", 0.0)
    dir_f1 = result.get("f1", 0.0)

    logging.info(f"   Result: Skel F1={skel_f1:.2f} | Dir F1={dir_f1:.2f}")
    safe_result = {k: (float(v) if v is not None else 0.0) for k, v in result.items()}
    return safe_result


def main(args):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("FEDCDH EVALUATION (New Architecture)")
    res_list = []

    for i in range(args.N):
        try:
            res = test_fedCDH(i, args)
            res_list.append(list(res.values()))
        except Exception as e:
            logging.error(f"Instance {i} failed: {e}", exc_info=True)

    if not res_list:
        return

    avg = np.mean(res_list, axis=0)
    std = np.std(res_list, axis=0)
    keys = list(res.keys())

    print("=" * 60)
    print(f"FINAL RESULTS ({args.scenario.upper()} - {args.ci_method.upper()})")
    print("Metrics:", keys)
    print("Average:", np.array2string(avg, precision=3, separator=", "))
    print("Std Dev:", np.array2string(std, precision=3, separator=", "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=1, type=int)
    parser.add_argument("--d", default=5, type=int)
    parser.add_argument("--K", default=2, type=int)
    parser.add_argument("--n", default=200, type=int)
    parser.add_argument("--model_type", default="general", type=str)
    parser.add_argument("--ci_method", default="spn", type=str)
    parser.add_argument("--scenario", default="horizontal", type=str)
    args = parser.parse_args()
    main(args)
