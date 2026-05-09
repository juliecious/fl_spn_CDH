"""Minimal test to debug aggregation strategy."""
import sys, os

sys.path.insert(0, os.path.abspath("."))
import logging

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

import numpy as np
from argparse import Namespace
from causallearn.search.FCMBased.FedCDH import FedCDH
from causallearn.utils.data_utils import (
    simulate_dag,
    simulate_parameter,
    my_simulate_linear_gaussian,
    set_random_seed,
)

set_random_seed(42)
d, K, n_total = 6, 2, 400  # Minimal config

# Generate data
G_bin = simulate_dag(d, d, "ER")
B = simulate_parameter(G_bin)
X_samples, _ = my_simulate_linear_gaussian(B, K, n_total, "gauss")
c_indx = np.zeros((n_total, 1))
n_per = n_total // K
X_splits = [X_samples[k * n_per : (k + 1) * n_per, :] for k in range(K)]

args = Namespace(
    K=K,
    d=d,
    scenario="horizontal",
    model_type="SPN",
    ci_method="spn",  # Changed from "SPN_CIT" to "spn"
    n=n_per,
    epochs=2,
    alpha=0.05,
    data_type="linear",
    device="cpu",
    horizontal_aggregation="structure_voting",
    structure_vote_threshold=0.5,
    skip_spn_eval=True,  # Skip expensive evaluation for smoke test
)

print(f"\n{'='*80}")
print(f"MINIMAL TEST: horizontal_aggregation={args.horizontal_aggregation}")
print(f"{'='*80}\n")

fedcdh = FedCDH(args)
print(f"After __init__: fedcdh.horizontal_aggregation={fedcdh.horizontal_aggregation}")
print(f"After __init__: fedcdh.scenario={fedcdh.scenario}")
print()

# Call fit
fedcdh.fit(X_splits, c_indx, G_bin)

print(f"\n{'='*80}")
print("AFTER FIT - Checking attributes:")
print(f"{'='*80}")
print(
    f"hasattr consensus_dependency_graph: {hasattr(fedcdh, 'consensus_dependency_graph')}"
)
print(f"hasattr edge_confidence: {hasattr(fedcdh, 'edge_confidence')}")
print(f"hasattr fed_spn_model: {hasattr(fedcdh, 'fed_spn_model')}")
if hasattr(fedcdh, "fed_spn_model"):
    print(f"fed_spn_model is None: {fedcdh.fed_spn_model is None}")
print(f"{'='*80}\n")
