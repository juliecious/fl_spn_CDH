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
    """
    Calibrates threshold using Mean + k*Std instead of Max/Percentile.
    Robust against single outliers in the noise.
    """
    scores = []
    n_samples, n_features = data_matrix.shape

    # Strategy selection (Same as before)
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

    # Mean + 3 Sigma (Robust)
    noise_mean = np.mean(abs_scores)
    noise_std = np.std(abs_scores)

    # 3.0 is standard for "99.7% confidence" in a Normal distribution
    calibrated_thresh = noise_mean + (sigma_multiplier * noise_std)

    logging.info(f"    Noise Mean: {noise_mean:.2f} | Std: {noise_std:.2f}")
    logging.info(
        f"    Suggested Threshold (Mean + {sigma_multiplier}*Std): {calibrated_thresh:.2f}"
    )

    # Safety clamp: Don't let it go below 2.5 (too noisy) or above 5.0 (too strict)
    final_threshold = np.clip(calibrated_thresh, 2.5, 5.0)

    return final_threshold


def get_spn_z_score(server, data_numpy, x_idx, y_idx):
    """
    Helper to extract the raw Z-score (Signal-to-Noise Ratio) from the SPN.
    This mimics the internal logic of the CI test but returns the float score instead of boolean.
    """
    # Convert to tensor
    data_t = torch.tensor(data_numpy, dtype=torch.float32).to(server.device)

    # 1. Calculate Observed CMI (Conditional Mutual Information)
    # Z is empty [] because we are calibrating pairwise independence
    # _calculate_cmi_value should be available in your ServerSPN class logic
    # If it's named differently (e.g. _calculate_cmi), adjust here.
    cmi_obs = server._calculate_cmi_value(data_t, [x_idx], [y_idx], [])

    # 2. Null Distribution (Permutation Test)
    # We do a quick permutation to find the "Zero Baseline" for this specific pair
    null_vals = []
    # 5 permutations is enough for a rough Z-score estimate during calibration
    for _ in range(5):
        data_perm = data_t.clone()
        perm_indices = torch.randperm(data_t.size(0))

        # Shuffle X against Y (keeping others fixed)
        data_perm[:, x_idx] = data_t[perm_indices, x_idx]

        val = server._calculate_cmi_value(data_perm, [x_idx], [y_idx], [])
        null_vals.append(max(0.0, val))  # CMI theoretically >= 0

    null_mean = np.mean(null_vals)
    null_std = np.std(null_vals)

    # Avoid division by zero
    if null_std < 1e-9:
        null_std = 1e-9

    # 3. Calculate Z-Score
    z_score = (cmi_obs - null_mean) / null_std
    return z_score


# Simulation
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
        f"Running Instance {i} | Client K={K_clients} | samples per K {n_samples_per_client} | features={d_features} | CI={ci_method} | Scenario={scenario} | Model={model_type}"
    )

    # 1. Data Generation (Global Ground Truth)
    true_DAG_bin = simulate_dag(d_features, s0, "ER")

    # Generate heterogenous data (returns X and domain_index)
    # Note: my_simulate_linear_gaussian generates [N*K, D]
    total_samples = n_samples_per_client * K_clients

    if model_type == "linear":
        X_global, c_indx = my_simulate_linear_gaussian(
            true_DAG_bin, K_clients, total_samples, "gauss"
        )
    else:
        # General functional model.
        X_global, c_indx = my_simulate_general_hetero(
            true_DAG_bin, K_clients, total_samples, "gauss"
        )

    c_indx = np.repeat(np.arange(K_clients), n_samples_per_client).reshape(-1, 1)

    # Data Normalization (Crucial for SPNs)
    X_global = (X_global - X_global.mean(0)) / (X_global.std(0) + 1e-6)

    logging.info(">>> Phase 1: Finding edges...")
    # 2. FEDERATED SETUP (Phase 1: Density Estimation)
    if ci_method == "spn":
        start_train = time.time()

        # Configure Scenario
        if scenario == "hybrid":
            num_clusters = 10
        elif scenario == "vertical":
            num_clusters = 5
        else:  # horizontal
            num_clusters = 1

        # Extract Proxy Data (Simulate Client 0's view)
        if scenario == "horizontal":
            # Client sees all features, but subset of rows
            # We take the first chunk of data
            chunk_size = X_global.shape[0] // K_clients
            proxy_data = X_global[:chunk_size]

        elif scenario == "vertical":
            # Client sees all rows, but subset of features
            # We take the first chunk of features
            feat_chunk = d_features // K_clients
            proxy_data = X_global[:, :feat_chunk]

        elif scenario == "hybrid":
            # Client sees subset rows AND subset features
            chunk_size = X_global.shape[0] // K_clients
            feat_chunk = d_features // 2  # Approximation of hybrid feature split
            proxy_data = X_global[:chunk_size, :feat_chunk]

        # Run Auto-Tuner to find optimal depth, leaves, lr, and batch_size
        best_params = auto_tune_spn_config(
            proxy_data, num_clusters=num_clusters, n_trials=5  # Keep low for speed
        )

        #  Server & Client Initialization ---
        server = ServerSPN(
            global_num_features=d_features,
            scenario=scenario,
            num_clusters=num_clusters,
            threshold=0.01,  # Placeholder
        )

        # Simulate Clients
        # Split Data based on Scenario
        if scenario == "horizontal":
            # Split samples (Rows)
            X_splits = np.array_split(X_global, K_clients)
            feat_indices = [
                list(range(d_features)) for _ in range(K_clients)
            ]  # All have all features

        elif scenario == "vertical":
            # Split features (Cols)
            # Simple partition: d/K features per client (assuming d divides K)
            feats_per_client = d_features // K_clients
            X_splits = []
            feat_indices = []
            for k in range(K_clients):
                start = k * feats_per_client
                end = (k + 1) * feats_per_client if k < K_clients - 1 else d_features
                cols = list(range(start, end))
                # Client gets ALL rows, but subset of cols
                X_splits.append(X_global[:, cols])
                feat_indices.append(cols)

        elif scenario == "hybrid":
            # Example: Split samples in half, and split features with some overlap
            # Split rows (Horizontal component)
            sample_splits = np.array_split(X_global, K_clients)
            X_splits = []
            feat_indices = []

            # Simple Hybrid Logic:
            # Client 0 gets first half of samples, features [0, 1, 2, 3]
            # Client 1 gets second half of samples, features [2, 3, 4, 5]
            mid_feat = d_features // 2
            for k in range(K_clients):
                if k == 0:
                    cols = list(range(0, mid_feat + 1))  # Features 0, 1, 2, 3
                else:
                    cols = list(range(mid_feat - 1, d_features))  # Features 2, 3, 4, 5

                X_splits.append(sample_splits[k][:, cols])
                feat_indices.append(cols)

        # Train Loop
        for k in range(K_clients):
            client = ClientSPN(
                client_id=k,
                num_features=X_splits[k].shape[1],
                num_clusters=num_clusters,
                **best_params,  # Unpacks: depth, num_sums, lr, batch_size, etc.
            )
            client.train(X_splits[k], epochs=30)

            # Register with Server
            server.register_client(client, feature_indices=feat_indices[k])

        train_time = time.time() - start_train
        print(f">>> Phase 1 Complete ({train_time:.2f}s)")

        # CDNOD queries indices up to d (the domain index).
        # We must provide a matrix that includes this column to prevent "out of bounds".
        data_aug = np.hstack([X_global, c_indx])

        calibrated_threshold = calibrate_threshold(
            server,
            data_aug,
            shuffles_per_pair=2,  # Fast calibration
            sigma_multiplier=3.5,  # The "Goldilocks" setting
        )
        server.threshold = calibrated_threshold
        suggested_threshold = calibrated_threshold

        # Define the Oracle Wrapper for CDNOD
        # This function signature must capture the 'server' object
        def oracle_wrapper(X_in, Y_in, Z_in=None, *args, **kwargs):
            # We must support the Z_in=None case for marginal independence
            if Z_in is None:
                Z_in = []

            # 10 perms is for high precision. 5 perms is acceptable for a CPU Demo.
            return server.ci_test(
                X_in,
                Y_in,
                Z_in,
                data_matrix=data_aug,
                num_permutations=2,
                sigma_threshold=suggested_threshold,
            )

        oracle_wrapper.method = "spn"
        indep_test_obj = oracle_wrapper

    else:
        logging.info(">>> Phase 1: Finding edges...")
        # Benchmark Methods (KCI, FisherZ)
        indep_test_obj = ci_method

    # 3. CAUSAL DISCOVERY (Phase 2)
    logging.info(">>> Phase 2: Causal Discovery (CDNOD)...")
    start_cd = time.time()

    # We pass the Wrapper Object if SPN, else string
    cg = cdnod(
        X_global,
        c_indx,
        K_clients,
        alpha=0.01,
        indep_test=indep_test_obj,  # <--- Passing our Federated Oracle
        stable=True,
        uc_rule=0,
        uc_priority=-1,
    )

    cd_time = time.time() - start_cd

    # 4. Evaluation
    est_graph = cg.G.graph[0:d_features, 0:d_features]  # Ignore C_indx node
    est_cpdag = get_cpdag_from_cdnod(est_graph)
    est_dag = get_dag_from_pdag(est_cpdag)

    res_skel = count_skeleton_accuracy(true_DAG_bin, est_cpdag)
    res_dir = count_dag_accuracy(true_DAG_bin, est_dag)

    result = {
        **res_skel,
        **res_dir,
        "time_train": train_time if ci_method == "spn" else 0,
        "time_cd": cd_time,
    }

    # Pretty Print
    logging.info(
        f"   Result: Skel F1={result['f1_skeleton']:.2f} | Dir F1={result['f1']:.2f}"
    )
    return result


def main(args):
    """
    Main evaluation function with CI method comparison
    """
    logging.info("=" * 60)
    logging.info("FEDCDH EVALUATION WITH CONFIGURABLE CI METHODS")
    logging.info("=" * 60)

    res_list = []
    successful_runs = 0

    for i in range(args.N):
        try:
            res = test_fedCDH(
                i=i,
                n_samples_per_client=args.n,
                K_clients=args.K,
                d_features=args.d,
                s0=args.d,
                model_type=args.model_type,
                ci_method=args.ci_method,
                scenario=args.scenario,
            )
            res_val = list(res.values())

            if None in res_val:
                logging.info(f"Warning: None values in results for instance {i}")
            else:
                res_list.append(res_val)
                successful_runs += 1

        except Exception as e:
            logging.info(f"Error in instance {i}: {e}")
            continue

    if successful_runs == 0:
        logging.info("No successful runs!")
        return

    # Compute statistics
    res_list = np.array(res_list)

    avg = np.mean(res_list, axis=0)
    std = np.std(res_list, axis=0)

    # Print results
    result_keys = [k for k in res.keys() if k != "ci_method"]
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Successful runs: {successful_runs}/{args.N}")
    print(f"CI Method: {args.ci_method.upper()}")
    print()
    print("Metrics:", result_keys)
    print("Average:", avg)
    print("Std Dev:", std)

    # Key metrics summary
    skeleton_f1_idx = next(i for i, k in enumerate(result_keys) if "f1_skeleton" in k)
    direction_f1_idx = next(i for i, k in enumerate(result_keys) if k == "f1")

    time_train_idx = next(i for i, k in enumerate(result_keys) if k == "time_train")
    time_cd_idx = next(i for i, k in enumerate(result_keys) if k == "time_cd")

    logging.info("SUMMARY:")
    logging.info(
        f"  Skeleton F1: {avg[skeleton_f1_idx]:.3f} ± {std[skeleton_f1_idx]:.3f}"
    )
    logging.info(
        f"  Direction F1: {avg[direction_f1_idx]:.3f} ± {std[direction_f1_idx]:.3f}"
    )
    logging.info(f"  Avg Train Time: {avg[time_train_idx]:.2f}s")
    logging.info(f"  Avg Discovery Time: {avg[time_cd_idx]:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated Causal Discovery with Configurable CI Tests"
    )

    # Existing parameters
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

    # New CI method selection parameter
    parser.add_argument(
        "--ci_method",
        default="spn",
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
