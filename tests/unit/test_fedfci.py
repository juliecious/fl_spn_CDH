import sys
import torch
import numpy as np
import time
import argparse
import logging
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.FedPC import (
    GlobalFedSPN,
    LocalSPNWrapper,
)
from causallearn.utils.cit import SPN_CIT
from causallearn.graph.GraphNode import GraphNode
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

np.set_printoptions(suppress=True, precision=3)
torch.set_num_threads(1)


def test_fedFCI(i, args):
    """
    Test Federated FCI (Fast Causal Inference) with Latent Confounders.
    Scenario: X <- L -> Y. L is unobserved.
    Expected: X <-> Y (Bi-directed edge in PAG).
    """
    logging.info(
        f"Instance {i} | K={args.K} | N={args.n} | Scenario={args.scenario} | Model=FCI"
    )

    np.random.seed(i)
    torch.manual_seed(i)

    # 1. Data Generation (Latent Confounder)
    # L ~ N(0, 1)
    # X = L + N(0, 0.5)
    # Y = L + N(0, 0.5)
    # We observe X, Y. We do NOT observe L.

    n_total = args.n * args.K
    L = np.random.normal(0, 1, n_total)
    X = L + np.random.normal(0, 0.5, n_total)
    Y = L + np.random.normal(0, 0.5, n_total)

    # Observed Data: [X, Y]
    # Indices: X=0, Y=1
    X_global = np.stack([X, Y], axis=1).astype(np.float32)
    d_features = 2

    # Simple Normalization
    X_global = (X_global - X_global.mean(0)) / (X_global.std(0) + 1e-6)

    logging.info(">>> Phase 1: Density Estimation (FedSPN)...")
    start_train = time.time()

    # Horizontal Split
    X_splits = np.array_split(X_global, args.K)
    feature_maps = {k: list(range(d_features)) for k in range(args.K)}

    local_models = []
    for k in range(args.K):
        # logging.info(f"   Training Client {k+1}/{args.K}...")
        leaf = LocalSPNWrapper(
            num_features=d_features,
            device="cpu",
            num_sums=20,
            num_leaves=20,
            depth=1,  # Shallow is enough for 2 vars
            num_repetitions=5,
            seed=i * 100 + k,
        )
        leaf.train_local(X_splits[k], epochs=30, lr=0.01)
        local_models.append(leaf)

    global_spn = GlobalFedSPN(
        local_models, device="cpu", feature_map=feature_maps, strategy="mixture"
    )
    train_time = time.time() - start_train
    logging.info(f"   FedSPN Training Complete ({train_time:.2f}s)")

    # 2. Causal Discovery (FCI)
    logging.info(">>> Phase 2: Causal Discovery (FedFCI)...")
    start_cd = time.time()

    # Initialize SPN Oracle
    # We pass data_matrix because SPN_CIT wrapper expects it for dimensionality checks,
    # though it queries the model.
    oracle = SPN_CIT(
        X_global, global_model=global_spn, threshold=0.01, num_permutations=100
    )

    # Run FCI
    # output: G (GeneralGraph), edges (list)
    G, edges = fci(X_global, oracle, alpha=0.01, verbose=False)

    cd_time = time.time() - start_cd

    # 3. Validation
    # We expect X (0) <-> Y (1).
    # In PAG:
    #   Circle (o): 1
    #   Arrow (>): 2
    #   Tail (-): 3 (not typically used in PAG for endpoints?)
    #   Null: 0

    # FCI Edge types:
    # 1: Circle (o)
    # 2: Arrow (>)
    # 3: Tail (-)

    # X <-> Y means:
    # G.graph[0, 1] == 2 (Arrow at Y)
    # G.graph[1, 0] == 2 (Arrow at X)

    edge_x_y = G.graph[0, 1]
    edge_y_x = G.graph[1, 0]

    logging.info(f"   Graph Edge X-Y: {edge_x_y} (2=Arrow, 1=Circle, 3=Tail)")
    logging.info(f"   Graph Edge Y-X: {edge_y_x} (2=Arrow, 1=Circle, 3=Tail)")

    # Success Criteria: Both are Arrows (2) or Circles (1) implying confounding/uncertainty.
    # Failure Criteria: Tail (3) at either end (implies X->Y or Y->X or X-Y).

    is_confounded = (edge_x_y in [1, 2]) and (edge_y_x in [1, 2])

    if is_confounded:
        logging.info("SUCCESS: FCI detected confounding (Bi-directed/Circle edge).")
    else:
        logging.error("FAILURE: FCI inferred unconfounded structure.")

    return {
        "is_confounded": is_confounded,
        "time_train": train_time,
        "time_cd": cd_time,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=1, type=int)
    parser.add_argument("--n", default=500, type=int)
    parser.add_argument("--K", default=2, type=int)
    parser.add_argument("--scenario", default="horizontal", type=str)
    args = parser.parse_args()

    test_fedFCI(0, args)
