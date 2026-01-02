import sys
import torch
import numpy as np
import time
import argparse
import logging
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import CIT, fisherz, kci
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

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_fedCDH(i, args):
    set_random_seed(i)
    logging.info(
        f"=== Instance {i} | K={args.K} | N={args.n} | Scen={args.scenario} | Model={args.model_type} ==="
    )

    # ---------------------------------------------------------
    # 1. Data Generation
    # ---------------------------------------------------------
    true_DAG_bin = simulate_dag(args.d, args.d, "ER")
    total_samples = args.n * args.K

    if args.model_type == "linear":
        X_global, _ = my_simulate_linear_gaussian(
            true_DAG_bin, args.K, total_samples, "gauss"
        )
    else:
        X_global, _ = my_simulate_general_hetero(
            true_DAG_bin, args.K, total_samples, "gauss"
        )

    X_global = (X_global - X_global.mean(0)) / (X_global.std(0) + 1e-6)

    # ---------------------------------------------------------
    # 2. Federated SPN Training
    # ---------------------------------------------------------
    logging.info(">>> Phase 1: Federated SPN Training...")
    start_train = time.time()

    if args.scenario == "hybrid":
        num_clusters = 5
    elif args.scenario == "vertical":
        num_clusters = 3
    else:
        num_clusters = 1

    # Auto-Tune
    proxy_data = X_global[: min(500, len(X_global))]
    if args.scenario != "horizontal":
        proxy_data = proxy_data[:, : args.d // 2]
    best_params = auto_tune_spn_config(
        proxy_data, num_clusters=num_clusters, n_trials=5
    )

    server = ServerSPN(args.d, args.scenario, num_clusters)

    # Register Clients
    X_splits = []
    feat_indices = []

    if args.scenario == "horizontal":
        X_splits = np.array_split(X_global, args.K)
        for k in range(args.K):
            feat_indices.append(list(range(args.d)))
            client = ClientSPN(k, args.d, num_clusters, **best_params)
            server.register_client(client, list(range(args.d)))

    elif args.scenario == "vertical":
        feats_per_client = args.d // args.K
        for k in range(args.K):
            cols = list(
                range(
                    k * feats_per_client,
                    (k + 1) * feats_per_client if k < args.K - 1 else args.d,
                )
            )
            feat_indices.append(cols)
            X_splits.append(X_global[:, cols])
            client = ClientSPN(k, len(cols), num_clusters, **best_params)
            server.register_client(client, cols)

    elif args.scenario == "hybrid":
        row_splits = np.array_split(X_global, args.K)
        mid = args.d // 2
        for k in range(args.K):
            cols = list(range(0, mid + 1)) if k == 0 else list(range(mid - 1, args.d))
            feat_indices.append(cols)
            X_splits.append(row_splits[k][:, cols])
            client = ClientSPN(k, len(cols), num_clusters, **best_params)
            server.register_client(client, cols)

    # Train Loop
    for r in range(5):
        responsibilities = None
        if args.scenario in ["vertical", "hybrid"]:
            responsibilities = server.perform_e_step(X_global, feat_indices)

        for k in range(args.K):
            weights = None
            if responsibilities is not None:
                if args.scenario == "hybrid":
                    chunk = len(X_global) // args.K
                    weights = responsibilities[k * chunk : (k + 1) * chunk]
                else:
                    weights = responsibilities
            server.clients[k].train_epoch(X_splits[k], weights=weights)

    train_time = time.time() - start_train

    # ---------------------------------------------------------
    # 3. Generative Causal Discovery (The Fix)
    # ---------------------------------------------------------
    logging.info(">>> Phase 2: Generative Discovery (CDNOD)...")
    start_cd = time.time()

    # A. Generate Data
    X_syn, C_syn = server.generate_global_synthetic_data(n_samples=2000)
    X_syn = X_syn.astype(np.float32)
    C_syn = C_syn.astype(int)

    # B. Combine for Initialization (Important for KCI)
    # CDNOD treats the context as the last column of the data
    data_aug = np.concatenate((X_syn, C_syn), axis=1)

    logging.info(f"    Generated Data Shape: {data_aug.shape}")

    # C. Configure Independence Test with WRAPPER
    if args.model_type == "general":
        logging.info("    Using KCI (Kernel) test...")
        method_name = "kci"
    else:
        logging.info("    Using FisherZ test...")
        # FisherZ function expects (data, x, y, z), so it matches cdnod signature directly.
        method_name = "fisherz"

    # D. Run CDNOD
    # Note: We pass X_syn and C_syn. cdnod will concatenate them internally.
    # This matches the indices we initialized KCI with (0..D-1 are X, D is C).
    cg = cdnod(X_syn, C_syn, args.K, 0.05, method_name, True, 0, -1)

    cd_time = time.time() - start_cd + train_time

    # ---------------------------------------------------------
    # 4. Evaluation
    # ---------------------------------------------------------
    est_graph = cg.G.graph[: args.d, : args.d]
    est_cpdag = get_cpdag_from_cdnod(est_graph)
    est_dag = get_dag_from_pdag(est_cpdag)

    res_skel = count_skeleton_accuracy(true_DAG_bin, est_cpdag)
    res_dir = count_dag_accuracy(true_DAG_bin, est_dag)

    logging.info(
        f"    Result: Skel F1={res_skel['f1_skeleton']:.2f} | Dir F1={res_dir['f1']:.2f}"
    )
    return list(res_skel.values()) + list(res_dir.values()) + [train_time, cd_time]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=1, type=int)
    parser.add_argument("--d", default=5, type=int)
    parser.add_argument("--K", default=2, type=int)
    parser.add_argument("--n", default=1_000, type=int)
    parser.add_argument(
        "--model_type",
        default="general",
        type=str,
        help="Data generation model: linear or general",
    )
    parser.add_argument(
        "--scenario",
        default="hybrid",
        type=str,
        help="Data split scenario: horizontal, vertical or hybrid",
    )

    args = parser.parse_args()

    results = []
    for i in range(args.N):
        try:
            res = test_fedCDH(i, args)
            results.append(res)
        except Exception as e:
            logging.error(f"Run {i} failed: {e}")

    if results:
        avg = np.mean(results, axis=0)
        print("\nFINAL RESULTS:")
        print(
            "Metrics: [F1_Skel, Prec, Rec, SHD, F1_Dir, Prec, Rec, SHD, T_Train, T_CD]"
        )
        print("Avg:", np.array2string(avg, precision=3))


if __name__ == "__main__":
    main()
