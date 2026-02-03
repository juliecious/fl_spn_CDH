import os

# 1. Allow multiple OpenMP runtimes (The specific fix for SIGSEGV 139)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. Prevent thread contention between Torch and Scikit-learn
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import torch

# 3. Restrict Torch to 1 thread to let Scikit-learn (KMeans) run safely
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)


import numpy as np
import time
import argparse
import logging
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.FedPC import FedPC
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


def test_fedCDH(i, args):
    set_random_seed(i)
    logging.info(
        f"Instance {i} | K={args.K} | N={args.n} | D={args.d} | Scen={args.scenario}"
    )

    # 1. Data Generation
    true_DAG_bin = simulate_dag(args.d, args.d, "ER")
    total_samples = args.n * args.K

    if args.model_type == "linear":
        X_global, c_indx = my_simulate_linear_gaussian(
            true_DAG_bin, args.K, total_samples, "gauss"
        )
    else:
        X_global, c_indx = my_simulate_general_hetero(
            true_DAG_bin, args.K, total_samples, "gauss"
        )

    c_indx = np.repeat(np.arange(args.K), args.n).reshape(-1, 1)
    X_global = (X_global - X_global.mean(0)) / (X_global.std(0) + 1e-6)

    # 2. Configure Independence Test (SPN Oracle vs KCI)
    train_time = 0

    if args.ci_method == "fedpc":
        logging.info(">>> Phase 1: Training Network-Aligned FedPC...")
        start_train = time.time()

        # Initialize splits and model based on scenario
        X_splits = []
        # num_clusters only matters for Vertical/Hybrid (Latent Variable)
        fed_pc = FedPC(args.scenario, args.d, args.K, num_clusters=5, device="cpu")
        spn_params = {
            "num_sums": 5,
            "num_leaves": 5,
            "num_repetitions": 1,
        }
        if args.scenario == "horizontal":
            # Horizontal: Clients have different rows, same features
            X_splits = np.array_split(X_global, args.K)
            for k in range(args.K):
                # Register all features (0..D) to every client
                fed_pc.register_client(k, list(range(args.d)), leaf_params=spn_params)

        elif args.scenario == "vertical":
            # Vertical: Clients have same rows, subset of features
            feats_per = args.d // args.K
            for k in range(args.K):
                start, end = (
                    k * feats_per,
                    (k + 1) * feats_per if k < args.K - 1 else args.d,
                )
                X_splits.append(X_global[:, start:end])
                fed_pc.register_client(
                    k, list(range(start, end)), leaf_params=spn_params
                )

        elif args.scenario == "hybrid":
            # Hybrid: Split rows first (Groups), then features (Vertical)
            # Simplified for CDH test: Treat as Vertical but with subset of rows?
            # Or usually Hybrid = Horizontal Groups of Vertical Clients.
            # For this test script, let's treat it as Vertical Partitioning for simplicity,
            # or you can implement the full Hybrid logic if your FedPC class supports it.
            # Falling back to Vertical logic for compactness:
            feats_per = args.d // args.K
            for k in range(args.K):
                start, end = (
                    k * feats_per,
                    (k + 1) * feats_per if k < args.K - 1 else args.d,
                )
                X_splits.append(X_global[:, start:end])
                fed_pc.register_client(
                    k, list(range(start, end)), leaf_params=spn_params
                )

        # Algorithm 1: One-Pass Training
        fed_pc.fit_one_pass(X_global, X_splits)
        train_time = time.time() - start_train

        # Define Oracle Wrapper
        logging.info(">>> Phase 2: Oracle Causal Discovery...")

        def oracle_wrapper(X, Y, Z, *args, **kwargs):
            # args/kwargs capture extra data passed by cdnod
            data = kwargs.get("data_matrix", X_global)

            # Use the trained FedPC to calculate CMI
            val = fed_pc.calculate_cmi(X, Y, Z, data)

            # Thresholding (Ideally calibrated, hardcoded here for compactness)
            threshold = 0.05
            return 0.0 if val > threshold else 1.0  # 0.0 = Dependent (Reject Null)

        oracle_wrapper.method = "fedpc"
        indep_test_obj = oracle_wrapper

    else:
        # Standard KCI / FisherZ
        indep_test_obj = args.ci_method

    # 3. Run Causal Discovery
    start_cd = time.time()
    cg = cdnod(
        X_global, c_indx, args.K, alpha=0.01, indep_test=indep_test_obj, stable=True
    )
    cd_time = time.time() - start_cd

    # 4. Evaluate
    est_graph = cg.G.graph[: args.d, : args.d]
    est_cpdag = get_cpdag_from_cdnod(est_graph)
    est_dag = get_dag_from_pdag(est_cpdag)

    res_skel = count_skeleton_accuracy(true_DAG_bin, est_cpdag)
    res_dir = count_dag_accuracy(true_DAG_bin, est_dag)

    result = {**res_skel, **res_dir, "time_train": train_time, "time_cd": cd_time}
    print(result)
    print(f"   Result: Skel F1={result['f1_skeleton']:.2f} | Dir F1={result['f1']:.2f}")
    return result


def main(args):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    results = []
    for i in range(args.N):
        try:
            res = test_fedCDH(i, args)
            results.append(list(res.values()))
        except Exception as e:
            logging.error(f"Instance {i} failed: {e}")

    if results:
        avg = np.mean(results, axis=0)
        print("=" * 60)
        print(f"FINAL RESULTS ({args.scenario.upper()} - {args.ci_method.upper()})")
        print(
            "Metrics: [F1_Skel, Prec, Rec, SHD, F1_Dir, Prec, Rec, SHD, T_Train, T_CD]"
        )
        print("Avg:", np.array2string(avg, precision=3, separator=", "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=1, type=int)
    parser.add_argument("--d", default=8, type=int)
    parser.add_argument("--K", default=2, type=int)
    parser.add_argument("--n", default=100, type=int)
    parser.add_argument("--model_type", default="general", type=str)
    parser.add_argument("--ci_method", default="fedpc", type=str)
    parser.add_argument("--scenario", default="vertical", type=str)
    args = parser.parse_args()
    main(args)
