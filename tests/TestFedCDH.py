import sys

sys.path.append("")
import numpy as np
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import kci
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

np.set_printoptions(suppress=True, precision=3)


# Simulation
def test_fedCHD(i, n, K, d, s0, model, ci_method="kci"):
    set_random_seed(i)
    print(f"Running instance {i} with CI method: {ci_method}")

    c_indx = np.asarray(list(range(K)))
    c_indx = np.repeat(c_indx, n)
    c_indx = np.reshape(c_indx, (n * K, 1))

    graph_type, sem_type = "ER", "gauss"
    true_DAG_bin = simulate_dag(d, s0, graph_type)  # ground-truth binary matrix

    if model == "linear":
        # Linear Gaussian model.
        X, _ = my_simulate_linear_gaussian(true_DAG_bin, K, n * K, sem_type)
    else:
        # General functional model.
        X, _ = my_simulate_general_hetero(true_DAG_bin, K, n * K, sem_type)

    # Select CI test method
    start = time.time()
    if ci_method.lower() == "spn":
        # Pass SPN-specific parameters
        cg = cdnod(
            X,
            c_indx,
            K,
            alpha=0.05,
            indep_test="spn",
            stable=True,
            uc_rule=0,
            uc_priority=-1,
            epochs=60,
            lr=0.01,
        )  # SPN-specific kwargs
    else:
        # Use traditional methods
        cg = cdnod(
            X,
            c_indx,
            K,
            alpha=0.05,
            indep_test=ci_method,
            stable=True,
            uc_rule=0,
            uc_priority=-1,
        )
    end = time.time()

    # Extract results
    est_graph = np.zeros((d, d))
    est_graph = cg.G.graph[0:d, 0:d]
    est_cpdag = get_cpdag_from_cdnod(
        est_graph
    )  # est_graph[i,j]=-1 & est_graph[j,i]=1  ->  est_graph_cpdag[i,j]=1
    est_dag_from_pdag = get_dag_from_pdag(
        est_cpdag
    )  # return a DAG from a PDAG in causaldag.

    # Undirected skeleton: F1, recall, precision, SHD
    ret_skeleton = count_skeleton_accuracy(true_DAG_bin, est_cpdag)

    # Directed graph: F1, recall, precision, SHD
    ret_diretion = count_dag_accuracy(true_DAG_bin, est_dag_from_pdag)

    # Combine results
    result = {}
    result.update(ret_skeleton)
    result.update(ret_diretion)
    result["time"] = end - start
    print("")

    return result


def main(args):
    """
    Main evaluation function with CI method comparison
    """
    print("=" * 60)
    print("FEDCDH EVALUATION WITH CONFIGURABLE CI METHODS")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  N={args.N}, d={args.d}, K={args.K}, n={args.n}")
    print(f"  Model: {args.model}, CI Method: {args.ci_method}")
    print()

    res_list = []
    successful_runs = 0

    for i in range(args.N):
        try:
            res = test_fedCHD(
                i, args.n, args.K, args.d, args.d, args.model, args.ci_method
            )
            res_val = list(res.values())

            if None in res_val:
                print(f"Warning: None values in results for instance {i}")
            else:
                res_list.append(res_val)
                successful_runs += 1

        except Exception as e:
            print(f"Error in instance {i}: {e}")
            continue

    if successful_runs == 0:
        print("No successful runs!")
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

    time_idx = next(i for i, k in enumerate(result_keys) if k == "time")

    print()
    print("SUMMARY:")
    print(f"  Skeleton F1: {avg[skeleton_f1_idx]:.3f} ± {std[skeleton_f1_idx]:.3f}")
    print(f"  Direction F1: {avg[direction_f1_idx]:.3f} ± {std[direction_f1_idx]:.3f}")
    print(f"  Average Runtime: {avg[time_idx]:.2f}s ± {std[time_idx]:.2f}s")


def main_without_args(N, d, K, n, model):
    res_list = []
    for i in range(N):
        res = test_fedCHD(i, n, K, d, d, model)  # a dictionary
        res_val = list(res.values())
        if None in res_val:
            print("Error! None in results!")
        else:
            res_list.append(res_val)
    res_list = np.array(res_list)
    avg = np.mean(res_list, axis=0)  # skeleton, orientation
    std = np.std(res_list, axis=0)

    print("########## Measurement: ", list(res.keys()))
    print("########## Average:     ", avg)
    print("########## Std:         ", std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated Causal Discovery with Configurable CI Tests"
    )

    # Existing parameters
    parser.add_argument("--N", default=10, type=int, help="Number of test instances")
    parser.add_argument("--d", default=6, type=int, help="Number of variables")
    parser.add_argument("--K", default=10, type=int, help="Number of federated clients")
    parser.add_argument(
        "--n", default=100, type=int, help="Number of samples per client"
    )
    parser.add_argument(
        "--model",
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

    args = parser.parse_args()
    main(args)

    # N = 10
    # d = 6
    # K = 10
    # n = 100
    # model = 'linear'
    # main_without_args(N, d, K, n, model)
