import sys
import torch

sys.path.append("")
import numpy as np
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.SPN import ServerSPN, ClientSPN, auto_tune_spn_config
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
import time
import argparse
import logging
import itertools

np.set_printoptions(suppress=True, precision=3)


def calibrate_threshold(
    server, data_matrix, shuffles_per_pair=4, sigma_multiplier=3.0, max_pairs=30
):
    scores = []
    n_samples, n_features = data_matrix.shape

    if n_features < 10:
        pairs_to_check = list(itertools.combinations(range(n_features), 2))
    else:
        pairs_to_check = [
            np.random.choice(n_features, 2, replace=False) for _ in range(max_pairs)
        ]

    logging.info(f">>> [Calibration] Testing {len(pairs_to_check)} pairs...")

    for idx_i, idx_j in pairs_to_check:
        for _ in range(shuffles_per_pair):
            decoy_matrix = data_matrix.copy()
            np.random.shuffle(decoy_matrix[:, idx_j])
            z_score = get_spn_z_score(server, decoy_matrix, idx_i, idx_j)
            scores.append(z_score)

    abs_scores = np.abs(scores)
    noise_mean = np.mean(abs_scores)
    noise_std = np.std(abs_scores)
    calibrated_thresh = noise_mean + (sigma_multiplier * noise_std)

    logging.info(f"    Noise Mean: {noise_mean:.2f} | Std: {noise_std:.2f}")
    logging.info(f"    Suggested Threshold: {calibrated_thresh:.2f}")

    return np.clip(calibrated_thresh, 2.5, 5.0)


def get_spn_z_score(server, data_numpy, x_idx, y_idx):
    data_t = torch.tensor(data_numpy, dtype=torch.float32).to(server.device)
    cmi_obs = server._calculate_cmi_value(data_t, [x_idx], [y_idx], [])

    null_vals = []
    for _ in range(5):
        data_perm = data_t.clone()
        perm_indices = torch.randperm(data_t.size(0))
        data_perm[:, x_idx] = data_t[perm_indices, x_idx]
        val = server._calculate_cmi_value(data_perm, [x_idx], [y_idx], [])
        null_vals.append(max(0.0, val))

    null_mean = np.mean(null_vals)
    null_std = np.std(null_vals) + 1e-9
    return (cmi_obs - null_mean) / null_std


def test_fedCDH(
    i,
    n_samples_per_client,
    K_clients,
    d_features,
    s0,
    model_type,
    ci_method,
    scenario="horizontal",
):
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

    c_indx = np.repeat(np.arange(K_clients), n_samples_per_client).reshape(-1, 1)
    X_global = (X_global - X_global.mean(0)) / (X_global.std(0) + 1e-6)

    logging.info(">>> Phase 1: Density Estimation (SPN Training)...")
    if ci_method == "spn":
        start_train = time.time()

        if scenario == "hybrid":
            num_clusters = 10
        elif scenario == "vertical":
            num_clusters = 5
        else:
            num_clusters = 1

        # Proxy Data extraction
        if scenario == "horizontal":
            proxy_data = X_global[: X_global.shape[0] // K_clients]
        elif scenario == "vertical":
            proxy_data = X_global[:, : d_features // K_clients]
        else:
            proxy_data = X_global[: X_global.shape[0] // K_clients, : d_features // 2]

        best_params = auto_tune_spn_config(
            proxy_data, num_clusters=num_clusters, n_trials=5
        )

        server = ServerSPN(
            global_num_features=d_features, scenario=scenario, num_clusters=num_clusters
        )

        # --- Split Data ---
        if scenario == "horizontal":
            X_splits = np.array_split(X_global, K_clients)
            feat_indices = [list(range(d_features)) for _ in range(K_clients)]
            for k in range(K_clients):
                client = ClientSPN(k, d_features, num_clusters, **best_params)
                client.data_count = X_splits[k].shape[0]  # Set for FedAvg weighting
                server.register_client(client, feat_indices[k])

        elif scenario == "vertical":
            feats_per_client = d_features // K_clients
            X_splits = []
            feat_indices = []
            for k in range(K_clients):
                start = k * feats_per_client
                end = (k + 1) * feats_per_client if k < K_clients - 1 else d_features
                cols = list(range(start, end))
                X_splits.append(X_global[:, cols])
                feat_indices.append(cols)
                # Client only knows its subset of features
                client = ClientSPN(k, len(cols), num_clusters, **best_params)
                server.register_client(client, cols)

        elif scenario == "hybrid":
            sample_splits = np.array_split(X_global, K_clients)
            X_splits = []
            feat_indices = []
            mid_feat = d_features // 2
            for k in range(K_clients):
                if k == 0:
                    cols = list(range(0, mid_feat + 1))
                else:
                    cols = list(range(mid_feat - 1, d_features))

                X_splits.append(sample_splits[k][:, cols])
                feat_indices.append(cols)
                client = ClientSPN(k, len(cols), num_clusters, **best_params)
                server.register_client(client, cols)

        # --- Federated Training Loop ---
        n_rounds = 10
        local_epochs = 3

        # Warm-up (Important for Vertical EM)
        logging.info(">>> Warm-starting clients locally...")
        for k in range(K_clients):
            server.clients[k].train_epoch(X_splits[k], lr=best_params["lr"])

        logging.info(f">>> Starting Federated Rounds ({scenario})...")
        for r in range(n_rounds):
            round_loss = 0

            # E-STEP: Vertical/Hybrid Alignment
            responsibilities = None
            if scenario in ["vertical", "hybrid"]:  # FIXED: Added hybrid
                responsibilities = server.perform_e_step(X_global, feat_indices)

            # M-STEP: Local Training
            for k in range(K_clients):
                # For vertical, responsibilities must be sliced if data was row-split (Hybrid)
                # But here X_global is used for E-step, so responsibilities align with global rows.
                # In Hybrid, X_splits[k] is a subset of rows. We must slice weights.

                curr_weights = responsibilities
                if scenario == "hybrid":
                    # This is a simulation simplification. In real FL, indices must be aligned.
                    # We assume X_splits follow the global order sequentially.
                    start_idx = k * (total_samples // K_clients)
                    end_idx = (k + 1) * (total_samples // K_clients)
                    if curr_weights is not None:
                        curr_weights = responsibilities[start_idx:end_idx]

                for _ in range(local_epochs):
                    loss = server.clients[k].train_epoch(
                        X_splits[k], weights=curr_weights, lr=best_params["lr"]
                    )
                    round_loss += loss

            # AGGREGATION: Horizontal Only
            if scenario == "horizontal":
                server.perform_fedavg()

            if r % 2 == 0:
                logging.info(f"   [Round {r}] Avg Loss: {round_loss / K_clients:.4f}")

        train_time = time.time() - start_train

        # Calibration
        data_aug = np.hstack([X_global, c_indx])
        calibrated_threshold = calibrate_threshold(server, data_aug)
        server.threshold = calibrated_threshold

        def oracle_wrapper(X_in, Y_in, Z_in=None, *args, **kwargs):
            if Z_in is None:
                Z_in = []
            return server.ci_test(
                X_in,
                Y_in,
                Z_in,
                data_matrix=data_aug,
                sigma_threshold=calibrated_threshold,
            )

        oracle_wrapper.method = "spn"

        indep_test_obj = oracle_wrapper
    else:
        indep_test_obj = ci_method
        train_time = 0

    # 3. Causal Discovery
    logging.info(">>> Phase 2: Causal Discovery...")
    start_cd = time.time()
    cg = cdnod(
        X_global,
        c_indx,
        K_clients,
        alpha=0.01,
        indep_test=indep_test_obj,
        stable=True,
        uc_rule=0,
        uc_priority=-1,
    )
    cd_time = time.time() - start_cd

    est_graph = cg.G.graph[0:d_features, 0:d_features]
    est_cpdag = get_cpdag_from_cdnod(est_graph)
    est_dag = get_dag_from_pdag(est_cpdag)

    res_skel = count_skeleton_accuracy(true_DAG_bin, est_cpdag)
    res_dir = count_dag_accuracy(true_DAG_bin, est_dag)

    result = {**res_skel, **res_dir, "time_train": train_time, "time_cd": cd_time}
    logging.info(
        f"   Result: Skel F1={result['f1_skeleton']:.2f} | Dir F1={result['f1']:.2f}"
    )
    return result


def main(args):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("FEDCDH EVALUATION")
    res_list = []

    for i in range(args.N):
        try:
            res = test_fedCDH(
                i,
                args.n,
                args.K,
                args.d,
                args.d,
                args.model_type,
                args.ci_method,
                args.scenario,
            )
            res_list.append(list(res.values()))
        except Exception as e:
            logging.error(f"Instance {i} failed: {e}")

    if not res_list:
        return

    avg = np.mean(res_list, axis=0)
    print("=" * 60)
    print(f"FINAL RESULTS ({args.scenario.upper()} - {args.ci_method.upper()})")
    print("Metrics:", list(res.keys()))
    print("Average:", avg)


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
        default="linear",
        type=str,
        help="Data generation model: linear or general",
    )
    parser.add_argument(
        "--ci_method",
        default="gsq",
        type=str,
        choices=["kci", "spn", "gsq"],
        help="Conditional independence test method: kci (traditional), gsq, or spn (Sum-Product Networks)",
    )

    parser.add_argument(
        "--scenario",
        default="horizontal",
        type=str,
        help="Data split scenario: horizontal, vertical or hybrid",
    )
    args = parser.parse_args()
    main(args)
