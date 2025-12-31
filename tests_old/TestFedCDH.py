import sys

sys.path.append("")
import numpy as np
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.SPN import ServerSPN, ClientSPN
from causallearn.utils.data_utils import (
    my_simulate_general_hetero,
    my_simulate_linear_gaussian,
    set_random_seed,
    simulate_dag,
)
from causallearn.utils.data_utils import count_skeleton_accuracy
from causallearn.utils.data_utils import (
    get_cpdag_from_cdnod,
    get_dag_from_pdag,
    count_dag_accuracy,
)
import time
import argparse
import logging

np.set_printoptions(suppress=True, precision=3)


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

    # 2. FEDERATED SETUP (Phase 1: Density Estimation)
    if ci_method == "spn":
        logging.info(">>> Phase 1: Federated Training...")
        start_train = time.time()

        if d_features >= 20:
            spn_config = {"depth": 5, "num_sums": 20, "num_leaves": 40}
        else:
            spn_config = {"depth": 3, "num_sums": 10, "num_leaves": 20}

        # Configure Scenario
        if scenario == "hybrid":
            num_clusters = 20  # Use latent clusters for vertical
            threshold_val = 0.01
        elif scenario == "vertical":
            num_clusters = 5
            threshold_val = 0.01
        else:
            num_clusters = 1  # Single mixture component for Horizontal
            threshold_val = 0.025

        # 2. TUNING: Lower Threshold
        server = ServerSPN(
            global_num_features=d_features,
            scenario=scenario,
            num_clusters=num_clusters,
            threshold=threshold_val,
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
                **spn_config,
            )
            client.train(X_splits[k], epochs=100, lr=0.05)

            # Register with Server
            server.register_client(client, feature_indices=feat_indices[k])

        train_time = time.time() - start_train
        print(f">>> Phase 1 Complete ({train_time:.2f}s)")

        # Define the Oracle Wrapper for CDNOD
        # This function signature must capture the 'server' object
        def oracle_wrapper(X_in, Y_in, Z_in=None, *args, **kwargs):
            # We must support the Z_in=None case for marginal independence
            if Z_in is None:
                Z_in = []
            # CDNOD queries indices up to d (the domain index).
            # We must provide a matrix that includes this column to prevent "out of bounds".
            data_aug = np.hstack([X_global, c_indx])
            # 10 perms is for high precision. 5 perms is acceptable for a CPU Demo.
            return server.ci_test(
                X_in,
                Y_in,
                Z_in,
                data_matrix=data_aug,
                num_permutations=5,
                sigma_threshold=3.0,
            )

        oracle_wrapper.method = "spn"
        indep_test_obj = oracle_wrapper

    else:
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
    parser.add_argument("--N", default=3, type=int, help="Number of test instances")
    parser.add_argument("--d", default=10, type=int, help="Number of variables")
    parser.add_argument("--K", default=2, type=int, help="Number of federated clients")
    parser.add_argument(
        "--n", default=500, type=int, help="Number of samples per client"
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
        default="kci",
        type=str,
        choices=["kci", "spn", "gsq"],
        help="Conditional independence test method: kci (traditional), gsq, or spn (Sum-Product Networks)",
    )

    parser.add_argument(
        "--scenario",
        default="vertical",
        type=str,
        help="Data split scenario: horizontal, vertical or hybrid",
    )

    args = parser.parse_args()
    main(args)
