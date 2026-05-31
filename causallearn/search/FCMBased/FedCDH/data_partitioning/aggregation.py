"""
Structure-Preserving Aggregation for Horizontal Federated Learning.

Implements majority voting on dependency graphs to preserve causal structure
during federated aggregation.

Key Functions:
- extract_local_dependency_graph: Extract dependencies from a local SPN via CI tests
- aggregate_structures_by_voting: Majority voting on edges from multiple local graphs
- build_structure_weighted_mixture: Create confidence-weighted mixture based on structure quality
"""

import logging
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional


def extract_local_dependency_graph(
    spn_model,
    X_data: np.ndarray,
    alpha: float = 0.05,
    num_permutations: int = 50,
    device: str = "cpu",
    has_augmented_var: bool = True,
) -> nx.Graph:
    """
    Extract dependency graph from a local SPN using conditional independence tests.

    Args:
        spn_model: Trained local SPN (LocalClusterMixture or LocalSPNWrapper)
        X_data: Training data [n, d] for this client
        alpha: Significance level for CI tests
        num_permutations: Number of permutations for SPN_CIT
        device: torch device
        has_augmented_var: If True, last column is augmented variable (condition on it)

    Returns:
        NetworkX Graph with edges representing dependencies (X⊥̸Y)

    Note:
        If has_augmented_var=True, we:
        1. Only test original variables (0 to d-2)
        2. Always condition on augmented variable (d-1)
        3. This preserves SPN context while excluding augmented var from causal graph
    """
    from causallearn.utils.cit import SPN_CIT

    d = X_data.shape[1]

    # Determine which variables to include in causal graph
    if has_augmented_var:
        # Last column is augmented variable - exclude from graph
        d_original = d - 1
        augmented_var_idx = d - 1
        logging.info(
            f"      → Conditioning on augmented variable (column {augmented_var_idx})"
        )
        logging.info(
            f"      → Testing only original {d_original} variables for causal structure"
        )
    else:
        d_original = d
        augmented_var_idx = None

    # Create SPN_CIT instance
    spn_cit = SPN_CIT(
        data=X_data,
        global_model=spn_model,
        threshold=alpha,
        num_permutations=num_permutations,
    )

    # Test all pairwise dependencies (skeleton) among ORIGINAL variables only
    dependency_graph = nx.Graph()
    dependency_graph.add_nodes_from(range(d_original))

    for i in range(d_original):
        for j in range(i + 1, d_original):
            # Test independence: X_i ⊥ X_j | augmented_var
            # This preserves SPN's 9D context while testing causal relationships
            if has_augmented_var:
                # Condition on augmented variable to preserve SPN context
                p_value = spn_cit(i, j, [augmented_var_idx])
            else:
                # No augmented variable - standard marginal test
                p_value = spn_cit(i, j, None)

            # If p_value <= alpha, reject independence → they are DEPENDENT
            if p_value <= alpha:
                dependency_graph.add_edge(i, j, p_value=p_value, weight=1.0 - p_value)

    return dependency_graph


def aggregate_structures_by_voting(
    local_graphs: List[nx.Graph],
    threshold: float = 0.5,
    min_votes: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[Tuple[int, int], float]]:
    """
    Aggregate multiple local dependency graphs using majority voting.

    Args:
        local_graphs: List of NetworkX graphs from each client
        threshold: Proportion of clients that must agree (0.5 = majority)
        min_votes: Minimum absolute number of votes (overrides threshold if provided)

    Returns:
        consensus_graph: NetworkX Graph with edges that meet the threshold
        edge_confidence: Dictionary mapping (i,j) -> confidence score [0,1]
    """
    K = len(local_graphs)

    if K == 0:
        return nx.Graph(), {}

    # Count votes for each edge
    edge_votes = defaultdict(int)
    edge_p_values = defaultdict(list)

    # Get all nodes (assume same feature set across clients in horizontal mode)
    d = max(max(g.nodes()) + 1 if len(g.nodes()) > 0 else 0 for g in local_graphs)

    for graph in local_graphs:
        for i, j in graph.edges():
            # Normalize edge tuple (always i < j)
            edge = (min(i, j), max(i, j))
            edge_votes[edge] += 1

            # Store p-value if available
            if "p_value" in graph[i][j]:
                edge_p_values[edge].append(graph[i][j]["p_value"])

    # Determine vote threshold
    if min_votes is not None:
        vote_threshold = min_votes
    else:
        vote_threshold = int(np.ceil(threshold * K))

    # Build consensus graph
    consensus_graph = nx.Graph()
    consensus_graph.add_nodes_from(range(d))
    edge_confidence = {}

    for edge, votes in edge_votes.items():
        confidence = votes / K
        edge_confidence[edge] = confidence

        if votes >= vote_threshold:
            i, j = edge
            # Use median p-value if available
            if edge in edge_p_values and len(edge_p_values[edge]) > 0:
                median_p = np.median(edge_p_values[edge])
                consensus_graph.add_edge(
                    i, j, votes=votes, confidence=confidence, p_value=median_p
                )
            else:
                consensus_graph.add_edge(i, j, votes=votes, confidence=confidence)

    return consensus_graph, edge_confidence


def compute_structure_quality_weights(
    local_spns: List,
    local_data: List[np.ndarray],
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute quality-based weights for each local SPN based on log-likelihood.

    Better SPNs (higher LL) are given higher weights in aggregation.

    Args:
        local_spns: List of trained local SPNs
        local_data: List of training data arrays for each client
        device: torch device

    Returns:
        quality_weights: Normalized weights [K] based on LL quality
    """
    import torch

    K = len(local_spns)
    quality_scores = []

    for spn, data in zip(local_spns, local_data):
        # Compute train log-likelihood
        with torch.no_grad():
            X_tensor = torch.from_numpy(data).float().to(device)
            ll = spn.log_prob(X_tensor).mean().item()

        # Convert to score (higher LL = better)
        # Use exp to amplify differences
        quality_scores.append(np.exp(ll / 10.0))  # Scale down to avoid overflow

    # Normalize to weights
    quality_weights = np.array(quality_scores)
    quality_weights = quality_weights / quality_weights.sum()

    return quality_weights


def build_structure_weighted_mixture(
    local_spns: List,
    local_data: List[np.ndarray],
    device: str = "cpu",
    weight_by_ll: bool = True,
):
    """
    Build global mixture with confidence-weighted mixing.

    Args:
        local_spns: List of local SPNs
        local_data: List of training data for each client
        device: torch device
        weight_by_ll: If True, weight by LL quality; if False, use uniform weights

    Returns:
        GlobalFedSPN with quality-based or uniform weights
    """
    from causallearn.utils.FedPC import GlobalFedSPN

    if weight_by_ll:
        weights = compute_structure_quality_weights(local_spns, local_data, device)
    else:
        K = len(local_spns)
        weights = np.ones(K) / K

    fed_spn = GlobalFedSPN(
        components=local_spns,
        weights=weights.tolist(),
        strategy="mixture",
        device=device,
    )

    return fed_spn, weights


def log_structure_aggregation_summary(
    local_graphs: List[nx.Graph],
    consensus_graph: nx.Graph,
    edge_confidence: Dict[Tuple[int, int], float],
    logger: Optional[logging.Logger] = None,
):
    """
    Log summary of structure aggregation process.

    Args:
        local_graphs: List of local dependency graphs
        consensus_graph: Aggregated consensus graph
        edge_confidence: Confidence scores for each edge
        logger: Logger instance (uses logging if None)
    """
    if logger is None:
        logger = logging

    K = len(local_graphs)

    # Count local edges
    local_edge_counts = [len(g.edges()) for g in local_graphs]

    logger.info("\n" + "=" * 60)
    logger.info("[Structure-Preserving Aggregation] Summary")
    logger.info("=" * 60)
    logger.info(f"Number of clients: {K}")
    logger.info(f"Local edge counts: {local_edge_counts}")
    logger.info(f"Average local edges: {np.mean(local_edge_counts):.1f}")
    logger.info(f"Consensus edges: {len(consensus_graph.edges())}")

    if len(consensus_graph.edges()) > 0:
        confidences = [
            edge_confidence.get(edge, 0.0) for edge in consensus_graph.edges()
        ]
        logger.info(f"Mean edge confidence: {np.mean(confidences):.3f}")
        logger.info(f"Min edge confidence: {np.min(confidences):.3f}")
        logger.info(f"Max edge confidence: {np.max(confidences):.3f}")

        # Show edges with high confidence
        high_conf_edges = [
            (i, j, edge_confidence[(i, j)])
            for (i, j) in consensus_graph.edges()
            if edge_confidence.get((i, j), 0) >= 0.75
        ]

        if high_conf_edges:
            logger.info(f"\nHigh-confidence edges (≥75% agreement):")
            for i, j, conf in sorted(high_conf_edges, key=lambda x: -x[2])[:10]:
                logger.info(f"  {i} -- {j}: {conf:.1%} confidence")
    else:
        logger.info("No consensus edges found (all votes below threshold)")

    logger.info("=" * 60 + "\n")
