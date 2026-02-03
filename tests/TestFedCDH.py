import sys
import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import logging
from tqdm import tqdm

sys.path.append("")

from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.FedPC import (
    FedPC,
    GlobalFedSPN,
    LocalSPNWrapper,
    auto_tune_spn_config,
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
    FedCDH passes data as [X, U].
    GlobalFedSPN models P(X).
    This wrapper models P(X, U) = P(U) * P(X | U).
    """

    def __init__(self, global_spn: GlobalFedSPN, u_index: int):
        super().__init__()
        self.spn = global_spn
        self.u_index = u_index
        self.device = global_spn.device

    def log_prob(self, x):
        """
        x: [Batch, D+1] tensor. Last column (or u_index) is U.
        """
        # 1. Split X and U
        if self.u_index == -1 or self.u_index == x.shape[1] - 1:
            x_feat = x[:, :-1]
            u_col = x[:, -1]
        else:
            # Assumes U is the specified index (usually last)
            x_feat = torch.cat([x[:, : self.u_index], x[:, self.u_index + 1 :]], dim=1)
            u_col = x[:, self.u_index]

        # 2. Check if U is observed (not NaN)
        # We check the first element (assuming batch consistency in masking)
        u_is_observed = not torch.isnan(u_col[0]).item()

        if u_is_observed:
            # Case: P(X, U=k) = P(U=k) * P(X | U=k)
            # P(X | U=k) -> LogProb from k-th client

            client_indices = u_col.long()

            # Compute log_prob for ALL clients on x_feat [Batch, K]
            # Optimization: could only compute for unique k in batch, but parallel is often faster
            client_lls = [c.log_prob(x_feat) for c in self.spn.clients]
            ll_stack = torch.cat(client_lls, dim=1)

            # Select the correct client for each sample
            # gather expects index to have same dims
            selected_ll = ll_stack.gather(1, client_indices.unsqueeze(1))  # [Batch, 1]

            # Add log weights: log P(U=k)
            weights = self.spn.weights.to(self.device)
            selected_weights = weights[client_indices].unsqueeze(1)

            return selected_ll + torch.log(selected_weights + 1e-9)

        else:
            # Case: P(X) = Sum_k P(U=k) P(X | U=k)
            # This is exactly what GlobalFedSPN.log_prob does
            return self.spn.log_prob(x_feat)

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

    set_random_seed(i)
    logging.info(
        f"Running Instance {i} | K={K_clients} | N_per_K={n_samples_per_client} | D={d_features} | Scenario={scenario}"
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

    # Simple global normalization for stability (though LocalSPNWrapper also normalizes)
    X_global = (X_global - X_global.mean(0)) / (X_global.std(0) + 1e-6)

    logging.info(">>> Phase 1: Density Estimation (FedSPN Training)...")
    train_time = 0
    fed_spn_model = None

    if ci_method == "spn":
        start_train = time.time()

        # Split data for Horizontal FL
        X_splits = np.array_split(X_global, K_clients)

        local_models = []

        # Train Local SPNs
        for k in range(K_clients):
            logging.info(f"   Training Client {k+1}/{K_clients}...")
            # Use Auto-Tune or defaults
            # For speed in test, use defaults but robust ones
            # Calculate safe depth
            safe_depth = max(1, int(np.floor(np.log2(d_features))))

            leaf = LocalSPNWrapper(
                num_features=d_features,
                device=device,
                num_sums=10,
                num_leaves=10,
                depth=safe_depth,
                num_repetitions=5,
                seed=i * 100 + k,  # Ensure structural heterogeneity per client
            )

            # Train
            loss = leaf.train_local(X_splits[k], epochs=30, lr=0.01)
            logging.info(f"      Client {k} Loss: {loss:.4f}")
            local_models.append(leaf)

        # Create Global SPN
        # Weights are uniform if sample sizes are equal
        global_spn = GlobalFedSPN(local_models, device=device)

        # Wrap for FedCDH (handling U index)
        # U is the last column (index d_features) in data_aug
        fed_spn_model = FedCDH_SPN_Wrapper(global_spn, u_index=d_features)

        train_time = time.time() - start_train
        logging.info(f"   FedSPN Training Complete ({train_time:.2f}s)")

    else:
        indep_test_obj = ci_method

    # 3. Causal Discovery
    logging.info(">>> Phase 2: Causal Discovery (FedCDH)...")
    start_cd = time.time()

    # Run CDNOD
    # Note: We pass fed_spn_model. If None, it falls back to 'fisherz' or whatever 'indep_test' is.
    cg = cdnod(
        X_global,
        c_indx,
        K_clients,
        alpha=0.01,
        indep_test=ci_method,
        stable=True,
        uc_rule=0,
        uc_priority=-1,
        fed_spn_model=fed_spn_model,
    )
    cd_time = time.time() - start_cd

    est_graph = cg.G.graph[0:d_features, 0:d_features]
    est_cpdag = get_cpdag_from_cdnod(est_graph)
    est_dag = get_dag_from_pdag(est_cpdag)

    res_skel = count_skeleton_accuracy(true_DAG_bin, est_cpdag)
    res_dir = count_dag_accuracy(true_DAG_bin, est_dag)

    result = {**res_skel, **res_dir, "time_train": train_time, "time_cd": cd_time}

    skel_f1 = result.get("f1_skeleton")
    if skel_f1 is None:
        skel_f1 = 0.0

    dir_f1 = result.get("f1")
    if dir_f1 is None:
        dir_f1 = 0.0

    logging.info(f"   Result: Skel F1={skel_f1:.2f} | Dir F1={dir_f1:.2f}")
    # Sanitize result for aggregation
    safe_result = {}
    for k, v in result.items():
        if v is None:
            safe_result[k] = 0.0
        else:
            safe_result[k] = float(v)

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

    # keys from the last result
    keys = list(res.keys()) if "res" in locals() else []

    print("=" * 60)
    print(f"FINAL RESULTS ({args.scenario.upper()} - {args.ci_method.upper()})")
    print("Metrics:", keys)
    print("Average:", np.array2string(avg, precision=3, separator=", "))
    print("Std Dev:", np.array2string(std, precision=3, separator=", "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=1, type=int, help="Number of test instances")
    parser.add_argument("--d", default=5, type=int, help="Number of variables")
    parser.add_argument("--K", default=2, type=int, help="Number of federated clients")
    parser.add_argument(
        "--n", default=200, type=int, help="Number of samples per client"
    )
    parser.add_argument(
        "--model_type",
        default="general",
        type=str,
        help="Data generation model: linear or general",
    )
    parser.add_argument(
        "--ci_method",
        default="spn",
        type=str,
        choices=["kci", "spn", "fisherz"],
        help="Conditional independence test method",
    )

    parser.add_argument(
        "--scenario",
        default="horizontal",
        type=str,
        help="Data split scenario: horizontal (default)",
    )

    args = parser.parse_args()
    main(args)
