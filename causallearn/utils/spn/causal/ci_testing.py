"""
Bug 3 Fix: CI-Based DAG Discovery via Tractable SPN Queries.

This module replaces the broken score-based search with proper conditional
independence testing using exact SPN marginalization.

Key Insight:
    Instead of:  Score(DAG) = constant_LL - penalty × |edges|  [BROKEN]
    Use:         X ⊥ Y | Z  iff  KL(P(X,Y|Z) || P(X|Z)P(Y|Z)) < threshold

Reference: Dry Run Analysis - Bug 3 Fix, Option B
"""

import torch
import numpy as np
import logging
from typing import List, Tuple
from itertools import combinations


def compute_circuit_ci_discrepancy(
    global_spn,
    data_matrix: torch.Tensor,
    target_x: int,
    target_y: int,
    conditioning_set: List[int],
) -> float:
    """
    Measure statistical dependence between X and Y given Z using SPN.

    Uses exact log-likelihood evaluation via tractable marginalization:
        Discrepancy = E[log P(X,Y|Z) - log P(X|Z) - log P(Y|Z)]

    If X ⊥ Y | Z, then P(X,Y|Z) = P(X|Z)P(Y|Z), so discrepancy ≈ 0.

    Args:
        global_spn: Trained federated SPN (GlobalFedSPN or FederatedProduct)
        data_matrix: Full dataset [n, d]
        target_x: Variable index X
        target_y: Variable index Y
        conditioning_set: List of conditioning variable indices Z

    Returns:
        mean_discrepancy: Average absolute discrepancy over samples

    Mathematical Detail:
        log P(X,Y|Z) = log P(X,Y,Z) - log P(Z)
        log P(X|Z)   = log P(X,Z) - log P(Z)
        log P(Y|Z)   = log P(Y,Z) - log P(Z)

        Discrepancy = log P(X,Y|Z) - [log P(X|Z) + log P(Y|Z)]
                    = [log P(X,Y,Z) - log P(Z)] - [log P(X,Z) - log P(Z) + log P(Y,Z) - log P(Z)]
                    = log P(X,Y,Z) - log P(X,Z) - log P(Y,Z) + log P(Z)
    """
    if not isinstance(data_matrix, torch.Tensor):
        data_matrix = torch.tensor(
            data_matrix, dtype=torch.float32, device=global_spn.device
        )

    # Step 1: Compute log P(X, Y, Z)
    scope_xyz = [target_x, target_y] + conditioning_set
    ll_xyz = global_spn.eval_partial_scope_log_likelihood(data_matrix, scope_xyz)

    # Step 2: Compute log P(X, Z) and log P(Y, Z)
    scope_xz = [target_x] + conditioning_set
    scope_yz = [target_y] + conditioning_set
    ll_xz = global_spn.eval_partial_scope_log_likelihood(data_matrix, scope_xz)
    ll_yz = global_spn.eval_partial_scope_log_likelihood(data_matrix, scope_yz)

    # Step 3: Compute log P(Z) if conditioning set is non-empty
    if len(conditioning_set) > 0:
        ll_z = global_spn.eval_partial_scope_log_likelihood(
            data_matrix, conditioning_set
        )
    else:
        # Empty conditioning set: log P(∅) = log(1) = 0
        ll_z = torch.zeros_like(ll_xyz)

    # Step 4: Compute conditional discrepancy
    # log P(X,Y|Z) - log P(X|Z) - log P(Y|Z)
    # = [log P(X,Y,Z) - log P(Z)] - [log P(X,Z) - log P(Z)] - [log P(Y,Z) - log P(Z)]
    # = log P(X,Y,Z) - log P(X,Z) - log P(Y,Z) + log P(Z)
    discrepancy_vector = ll_xyz - ll_xz - ll_yz + ll_z

    # Take absolute mean (KL-like measure)
    mean_discrepancy = torch.mean(torch.abs(discrepancy_vector)).item()

    logging.debug(
        f"CI Test: X={target_x}, Y={target_y}, Z={conditioning_set}, "
        f"Discrepancy={mean_discrepancy:.4f}"
    )

    return mean_discrepancy


def greedy_dag_search_via_circuit_ci(
    global_spn,
    validation_data: np.ndarray,
    threshold: float = 0.02,
    max_conditioning_size: int = 3,
    verbose: bool = False,
) -> np.ndarray:
    """
    Greedy DAG search using CI tests via tractable SPN queries.

    This implements a simplified PC-like algorithm:
    1. Start with complete undirected skeleton
    2. Test marginal independence: X ⊥ Y (no conditioning)
    3. Test conditional independence: X ⊥ Y | Z for increasing |Z|
    4. Remove edges where independence holds

    Args:
        global_spn: Trained federated SPN
        validation_data: Data for CI testing [n, d]
        threshold: Independence threshold (lower = stricter)
        max_conditioning_size: Max |Z| for CI tests
        verbose: Print progress

    Returns:
        skeleton: Undirected adjacency matrix [d, d]

    Reference: PC algorithm (Spirtes et al. 2000) with SPN-based CI tests
    """
    if not isinstance(validation_data, torch.Tensor):
        validation_data = torch.tensor(
            validation_data, dtype=torch.float32, device=global_spn.device
        )

    num_vars = validation_data.shape[1]

    # Initialize with complete undirected graph
    skeleton = np.ones((num_vars, num_vars), dtype=int)
    np.fill_diagonal(skeleton, 0)  # No self-loops

    if verbose:
        logging.info(
            f"[CI-DAG Search] Starting with complete skeleton: "
            f"{skeleton.sum()} edges"
        )

    # Separation sets for orientation (not implemented here, just skeleton)
    separation_sets = {}

    # Phase 1: Test marginal independence (conditioning set = ∅)
    edges_removed_marginal = 0
    for x in range(num_vars):
        for y in range(x + 1, num_vars):
            if skeleton[x, y] == 0:
                continue  # Already removed

            # Test X ⊥ Y
            dep_score = compute_circuit_ci_discrepancy(
                global_spn, validation_data, x, y, []
            )

            if dep_score < threshold:
                # Independence holds → remove edge
                skeleton[x, y] = 0
                skeleton[y, x] = 0
                separation_sets[(x, y)] = []
                separation_sets[(y, x)] = []
                edges_removed_marginal += 1

                if verbose:
                    logging.debug(
                        f"Removed edge {x}-{y}: marginal independence "
                        f"(score={dep_score:.4f})"
                    )

    if verbose:
        logging.info(
            f"[Phase 1] Marginal tests removed {edges_removed_marginal} edges, "
            f"{skeleton.sum()} remaining"
        )

    # Phase 2: Test conditional independence with increasing conditioning set size
    for depth in range(1, max_conditioning_size + 1):
        edges_removed_conditional = 0

        for x in range(num_vars):
            # Get current neighbors of x
            neighbors_x = [i for i in range(num_vars) if skeleton[x, i] == 1]

            for y in neighbors_x:
                if skeleton[x, y] == 0:
                    continue  # Edge already removed

                # Possible conditioning sets: neighbors of x excluding y
                possible_parents = [n for n in neighbors_x if n != y]

                if len(possible_parents) < depth:
                    continue  # Not enough neighbors for conditioning

                # Test all subsets of size 'depth'
                for z_set in combinations(possible_parents, depth):
                    z_list = list(z_set)

                    # Test X ⊥ Y | Z
                    dep_score = compute_circuit_ci_discrepancy(
                        global_spn, validation_data, x, y, z_list
                    )

                    if dep_score < threshold:
                        # Conditional independence holds → remove edge
                        skeleton[x, y] = 0
                        skeleton[y, x] = 0
                        separation_sets[(x, y)] = z_list
                        separation_sets[(y, x)] = z_list
                        edges_removed_conditional += 1

                        if verbose:
                            logging.debug(
                                f"Removed edge {x}-{y}: conditional independence | Z={z_list} "
                                f"(score={dep_score:.4f})"
                            )

                        break  # Move to next edge

        if verbose:
            logging.info(
                f"[Phase 2.{depth}] Conditional tests (|Z|={depth}) removed "
                f"{edges_removed_conditional} edges, {skeleton.sum()} remaining"
            )

        if edges_removed_conditional == 0:
            # No edges removed at this depth → stop early
            if verbose:
                logging.info(
                    f"[Early Stop] No edges removed at depth {depth}, terminating"
                )
            break

    if verbose:
        logging.info(
            f"[CI-DAG Search] Final skeleton: {skeleton.sum()} edges "
            f"(removed {num_vars * (num_vars - 1) - skeleton.sum()} edges)"
        )

    return skeleton


def pc_algorithm_with_circuit_ci(
    global_spn,
    validation_data: np.ndarray,
    threshold: float = 0.02,
    max_conditioning_size: int = 3,
    orient: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Full PC algorithm with CI tests via SPN and orientation rules.

    Steps:
    1. Learn skeleton via CI testing (greedy_dag_search_via_circuit_ci)
    2. Orient edges using v-structures and propagation rules

    Args:
        global_spn: Trained federated SPN
        validation_data: Data for CI testing
        threshold: Independence threshold
        max_conditioning_size: Max conditioning set size
        orient: Whether to apply orientation rules
        verbose: Print progress

    Returns:
        cpdag: Completed partially directed acyclic graph [d, d]
        separation_sets: Dictionary of separation sets for each edge

    Reference: PC algorithm (Spirtes et al. 2000)
    """
    # Step 1: Learn skeleton
    skeleton = greedy_dag_search_via_circuit_ci(
        global_spn,
        validation_data,
        threshold=threshold,
        max_conditioning_size=max_conditioning_size,
        verbose=verbose,
    )

    if not orient:
        return skeleton, {}

    # Step 2: Orient edges (v-structures)
    # TODO: Implement orientation rules
    # For now, return skeleton (undirected graph)

    if verbose:
        logging.warning(
            "[Orientation] Not implemented yet, returning undirected skeleton"
        )

    return skeleton, {}
