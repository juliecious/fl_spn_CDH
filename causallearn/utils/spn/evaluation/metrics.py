"""
SPN Quality Evaluation Metrics.

This module provides MMD, KS tests, and quality evaluation metrics for SPNs.
"""

"""
SPN Quality Evaluation Utilities.
Provides MMD, KS tests, and UMAP visualization for SPNs.
"""

import logging
import numpy as np
import torch
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances

# Try to import UMAP and matplotlib
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.debug("UMAP not available. Install with: pip install umap-learn")

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.debug("Matplotlib not available.")


# Gap 3 Fix: Fast Score-Based DAG Evaluation
def compute_fast_circuit_causal_score(
    global_pc_model,
    data: torch.Tensor,
    graph_edges_count: int,
    penalty: float = 1.0,
) -> float:
    """
    Compute BIC-style structural score using tractable SPN log-likelihood.

    This replaces expensive iterative CI tests with analytical log-likelihood
    evaluation, enabling scalable causal search on large graphs.

    Mathematical Form:
        Score = LL(Data | Circuit) - penalty × (|Edges|/2) × log(N)

    Where:
        - LL: Tractable log-likelihood from SPN forward pass
        - |Edges|: Number of edges in candidate DAG
        - N: Sample size
        - penalty: Scaling factor (default 1.0 = standard BIC)

    Key Advantages:
        1. O(log d) inference for SPNs vs O(exp(d)) for Bayesian Networks
        2. Single forward pass vs thousands of CI tests
        3. Parallelizable across GPU for large datasets

    Args:
        global_pc_model: Trained GlobalFedSPN or FederatedProduct
        data: Full dataset or validation subset [n, d]
        graph_edges_count: Number of edges in candidate DAG
        penalty: BIC penalty weight (higher = sparser graphs)

    Returns:
        score: Penalized log-likelihood (higher = better)

    Reference: Gap Analysis - Tractability Deficit in Score-Based Evaluation

    Example:
        >>> # Compare two DAG candidates
        >>> score_dag1 = compute_fast_circuit_causal_score(spn, data, edges=8)
        >>> score_dag2 = compute_fast_circuit_causal_score(spn, data, edges=12)
        >>> if score_dag1 > score_dag2:
        ...     print("DAG1 is better (simpler and fits data)")
    """
    import math

    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32, device=global_pc_model.device)

    total_samples = data.shape[0]

    # Step 1: Compute exact log-likelihood via tractable SPN
    with torch.no_grad():
        # FPC provides instantaneous log-likelihood evaluation
        log_probs = global_pc_model.log_prob(data)  # [n, 1]
        total_log_likelihood = torch.sum(log_probs).item()

    # Step 2: Compute BIC penalty for structural complexity
    # BIC = -2 × LL + k × log(N)
    # Rearranged: Score = LL - (k/2) × log(N) × penalty
    bic_penalty = 0.5 * graph_edges_count * math.log(total_samples)
    global_structural_score = total_log_likelihood - (penalty * bic_penalty)

    logging.debug(
        f"[ScoreBased] LL={total_log_likelihood:.3f}, "
        f"Edges={graph_edges_count}, "
        f"Penalty={bic_penalty:.3f}, "
        f"Score={global_structural_score:.3f}"
    )

    return global_structural_score


def greedy_dag_search_via_scores(
    global_spn,
    validation_data: np.ndarray,
    max_iterations: int = 100,
    penalty: float = 1.0,
    early_stop_threshold: float = 1e-2,
) -> np.ndarray:
    """
    Greedy hill-climbing search for optimal DAG using SPN scores.

    This replaces constraint-based skeleton search with score-based optimization,
    leveraging tractable SPN inference for scalability.

    Algorithm:
        1. Start with empty DAG (no edges)
        2. For each iteration:
            a. Try adding each possible edge (i → j)
            b. Compute score improvement via compute_fast_circuit_causal_score
            c. Select edge with highest improvement
            d. Check acyclicity constraint
        3. Stop when improvement < threshold

    Args:
        global_spn: Trained federated SPN
        validation_data: Data for scoring [n, d]
        max_iterations: Max search steps
        penalty: BIC penalty weight
        early_stop_threshold: Min score improvement to continue

    Returns:
        best_dag: Learned adjacency matrix [d, d] where A[i,j]=1 means i→j

    Reference: Gap Analysis - Step-by-Step Blueprint, Step 3
    """
    if not isinstance(validation_data, torch.Tensor):
        validation_data = torch.tensor(
            validation_data, dtype=torch.float32, device=global_spn.device
        )

    d = validation_data.shape[1]
    current_dag = np.zeros((d, d), dtype=int)
    current_score = compute_fast_circuit_causal_score(
        global_spn, validation_data, current_dag.sum(), penalty
    )

    logging.info(
        f"[DAGSearch] Starting greedy search, initial score: {current_score:.3f}"
    )

    for iteration in range(max_iterations):
        best_improvement = 0
        best_edge = None

        # Try all possible edge additions
        for i in range(d):
            for j in range(d):
                if i == j or current_dag[i, j] == 1:
                    continue  # Skip self-loops and existing edges

                # Tentatively add edge
                new_dag = current_dag.copy()
                new_dag[i, j] = 1

                # Check acyclicity
                if _is_cyclic(new_dag):
                    continue

                # Compute score
                new_score = compute_fast_circuit_causal_score(
                    global_spn, validation_data, new_dag.sum(), penalty
                )

                improvement = new_score - current_score

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_edge = (i, j)

        # Check stopping criteria
        if best_improvement < early_stop_threshold:
            logging.info(
                f"[DAGSearch] Converged at iteration {iteration} "
                f"(improvement={best_improvement:.4f})"
            )
            break

        # Add best edge
        if best_edge is not None:
            i, j = best_edge
            current_dag[i, j] = 1
            current_score += best_improvement

            logging.info(
                f"[Iteration {iteration}] Added edge {i}→{j}, "
                f"Score: {current_score:.3f} (+{best_improvement:.3f})"
            )

    logging.info(
        f"[DAGSearch] Final score: {current_score:.3f}, Edges: {current_dag.sum()}"
    )

    return current_dag


def _is_cyclic(adjacency_matrix: np.ndarray) -> bool:
    """
    Check if directed graph has cycles using DFS.

    Args:
        adjacency_matrix: [d, d] where A[i,j]=1 means edge i→j

    Returns:
        has_cycle: True if graph contains at least one cycle
    """
    d = adjacency_matrix.shape[0]
    visited = [False] * d
    rec_stack = [False] * d

    def dfs(node):
        visited[node] = True
        rec_stack[node] = True

        for neighbor in range(d):
            if adjacency_matrix[node, neighbor] == 1:
                if not visited[neighbor]:
                    if dfs(neighbor):
                        return True
                elif rec_stack[neighbor]:
                    return True

        rec_stack[node] = False
        return False

    for node in range(d):
        if not visited[node]:
            if dfs(node):
                return True
    return False


def compute_mmd_squared(X, Y, gamma=None):
    """
    Compute Maximum Mean Discrepancy (MMD²) using RBF kernel.

    Args:
        X: Real data [n, d]
        Y: Generated data [m, d]
        gamma: RBF kernel bandwidth (None = median heuristic)

    Returns:
        mmd_sq: MMD² statistic (non-negative)
    """
    n, m = len(X), len(Y)

    if n == 0 or m == 0:
        return 0.0

    # Median heuristic for bandwidth
    if gamma is None:
        combined = np.vstack([X[: min(100, n)], Y[: min(100, m)]])
        pairwise_dists = pairwise_distances(combined, combined)
        median_dist = np.median(pairwise_dists[pairwise_dists > 0])
        gamma = 1.0 / (2 * median_dist**2) if median_dist > 0 else 1.0

    # Compute kernel matrices
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)

    # Unbiased MMD² estimator
    mmd_sq = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) if n > 1 else 0
    mmd_sq += (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) if m > 1 else 0
    mmd_sq -= 2 * Kxy.mean()

    return max(0, mmd_sq)  # Ensure non-negative


def mmd_permutation_test(X, Y, n_permutations=50, gamma=None):
    """
    Permutation test for MMD to compute p-value.

    Args:
        X: Real data [n, d]
        Y: Generated data [m, d]
        n_permutations: Number of permutations
        gamma: RBF kernel bandwidth

    Returns:
        mmd_sq: Observed MMD²
        p_value: P-value from permutation test
    """
    mmd_obs = compute_mmd_squared(X, Y, gamma=gamma)

    combined = np.vstack([X, Y])
    n = len(X)
    m = len(Y)

    null_distribution = []
    for _ in range(n_permutations):
        perm_idx = np.random.permutation(len(combined))
        X_perm = combined[perm_idx[:n]]
        Y_perm = combined[perm_idx[n : n + m]]
        mmd_null = compute_mmd_squared(X_perm, Y_perm, gamma=gamma)
        null_distribution.append(mmd_null)

    null_distribution = np.array(null_distribution)
    p_value = (null_distribution >= mmd_obs).sum() / n_permutations

    return mmd_obs, p_value


def ks_test_dimensions(X_real, X_gen, alpha=0.05):
    """
    Kolmogorov-Smirnov test for each dimension with Bonferroni correction.

    Args:
        X_real: Real data [n, d]
        X_gen: Generated data [m, d]
        alpha: Significance level

    Returns:
        failed_dims: List of failed dimension indices
        ks_stats: KS statistics per dimension
        ks_pvalues: P-values per dimension
    """
    d = X_real.shape[1]
    failed_dims = []
    ks_stats = []
    ks_pvalues = []

    # Bonferroni correction
    alpha_corrected = alpha / d if d > 0 else alpha

    for dim in range(d):
        stat, pval = ks_2samp(X_real[:, dim], X_gen[:, dim])
        ks_stats.append(stat)
        ks_pvalues.append(pval)

        if pval < alpha_corrected:
            failed_dims.append(dim)

    return failed_dims, ks_stats, ks_pvalues


def evaluate_spn_quality(
    spn_model,
    X_data,
    n_samples=200,
    device="cpu",
    compute_mmd=True,
    compute_ks=True,
    name="SPN",
    has_context_column=True,
):
    """
    Evaluate SPN quality with MMD and KS tests.

    Args:
        spn_model: Trained SPN model
        X_data: Training data (with context column if applicable)
        n_samples: Number of samples to generate
        device: torch device
        compute_mmd: Whether to compute MMD
        compute_ks: Whether to compute KS test
        name: Name for logging
        has_context_column: If True, removes last column as context (default: True)

    Returns:
        Dictionary with metrics
    """
    results = {"name": name}

    try:
        # Log-likelihood on training data
        X_torch = torch.tensor(X_data, dtype=torch.float32).to(device)
        with torch.no_grad():
            train_ll = spn_model.log_prob(X_torch).mean().item()
        results["train_ll"] = train_ll

        # Sample from SPN
        with torch.no_grad():
            samples = spn_model.sample(n_samples).cpu().numpy()

        # Remove context column (last column) from both if present
        # BUGFIX: Only remove context column if has_context_column=True
        # In vertical/hybrid modes, global SPN doesn't have context column
        if has_context_column:
            X_features = X_data[:, :-1] if X_data.shape[1] > 1 else X_data
            samples_features = samples[:, :-1] if samples.shape[1] > 1 else samples
        else:
            X_features = X_data
            samples_features = samples

        # Ensure dimensions match
        if X_features.shape[1] != samples_features.shape[1]:
            logging.warning(
                f"{name}: Dimension mismatch: data={X_features.shape[1]}, samples={samples_features.shape[1]}"
            )
            return results

        d_features = X_features.shape[1]

        # MMD test
        if compute_mmd and d_features > 0:
            mmd_sq, mmd_pval = mmd_permutation_test(
                X_features, samples_features, n_permutations=50
            )
            results["mmd_squared"] = mmd_sq
            results["mmd_pvalue"] = mmd_pval

        # KS test
        if compute_ks and d_features > 0:
            failed_dims, ks_stats, ks_pvals = ks_test_dimensions(
                X_features, samples_features
            )
            results["ks_failed_dims"] = failed_dims
            results["ks_fail_ratio"] = (
                len(failed_dims) / d_features if d_features > 0 else 0
            )

    except Exception as e:
        logging.warning(f"Error evaluating {name}: {e}")

    return results


def create_simple_umap_plot(X_real, X_gen, save_path=None, title="UMAP Projection"):
    """
    Create UMAP visualization comparing real and generated data.

    Args:
        X_real: Real data [n, d]
        X_gen: Generated data [m, d]
        save_path: Path to save plot (None = don't save)
        title: Plot title

    Returns:
        embedding: UMAP embedding (or None if failed)
    """
    if not UMAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        return None

    if X_real.shape[1] < 2:
        logging.debug(f"UMAP skipped: need d>=2, got d={X_real.shape[1]}")
        return None

    try:
        # Combine data
        combined = np.vstack([X_real, X_gen])
        labels = np.array(["Real"] * len(X_real) + ["Generated"] * len(X_gen))

        # UMAP projection
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=min(15, len(combined) - 1),
            min_dist=0.1,
            verbose=False,
        )
        embedding = reducer.fit_transform(combined)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))

        real_mask = labels == "Real"
        ax.scatter(
            embedding[real_mask, 0],
            embedding[real_mask, 1],
            c="blue",
            alpha=0.5,
            s=20,
            label="Real Data",
        )
        ax.scatter(
            embedding[~real_mask, 0],
            embedding[~real_mask, 1],
            c="red",
            alpha=0.5,
            s=20,
            label="Generated Data",
        )

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logging.info(f"  Saved UMAP: {save_path}")

        plt.close()
        return embedding

    except Exception as e:
        logging.debug(f"UMAP visualization failed: {e}")
        return None


def evaluate_spn_independence_structure(
    spn_model,
    X_data,
    true_DAG_bin,
    alpha=0.05,
    max_order=1,
    n_conditional_tests=30,
    num_permutations=50,
    device="cpu",
    name="SPN",
):
    """
    Evaluate if SPN preserves independence structure from ground truth DAG.

    Uses existing SPN_CIT class to test conditional independencies.
    Compares SPN's CI test results with ground truth d-separation.

    Args:
        spn_model: Trained SPN (LocalSPNWrapper or GlobalFedSPN)
        X_data: Training data [n, d] (with context column if applicable)
        true_DAG_bin: Binary adjacency matrix [d, d] from simulate_dag()
        alpha: Significance level for CI test
        max_order: Max conditioning set size (0=skeleton, 1=order-1, etc.)
        n_conditional_tests: Number of conditional tests to sample
        num_permutations: Number of permutations for SPN_CIT
        device: torch device
        name: Name for logging

    Returns:
        Dictionary with independence structure metrics
    """
    import networkx as nx
    import random
    from causallearn.utils.cit import SPN_CIT
    from causallearn.utils.metrics import count_precision_recall_f1

    results = {"name": name}

    try:
        # Create SPN_CIT instance
        spn_cit = SPN_CIT(
            data=X_data,
            global_model=spn_model,
            threshold=alpha,
            num_permutations=num_permutations,
        )

        # Extract independence tests from ground truth DAG
        d = true_DAG_bin.shape[0]
        dag = nx.DiGraph(true_DAG_bin)
        test_cases = []

        # Order-0: Skeleton tests (all pairs, unconditional)
        all_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        for i, j in all_pairs:
            is_indep = nx.d_separated(dag, {i}, {j}, set())
            test_cases.append((i, j, tuple(), is_indep))

        # Order-1+: Conditional independence tests (sampled)
        if max_order >= 1 and d >= 3:
            for _ in range(n_conditional_tests):
                all_nodes = list(range(d))
                random.shuffle(all_nodes)
                X_var, Y_var = all_nodes[0], all_nodes[1]
                z_size = random.randint(1, min(max_order, d - 2))
                Z_vars = tuple(sorted(all_nodes[2 : 2 + z_size]))
                is_indep = nx.d_separated(dag, {X_var}, {Y_var}, set(Z_vars))
                test_cases.append((X_var, Y_var, Z_vars, is_indep))

        # Test each independence using SPN_CIT
        test_results = []
        for X_var, Y_var, Z_vars, gt_independent in test_cases:
            p_value = spn_cit(X_var, Y_var, list(Z_vars) if Z_vars else None)
            spn_independent = p_value > alpha
            test_results.append(
                (X_var, Y_var, Z_vars, gt_independent, spn_independent, p_value)
            )

        # Calculate confusion matrix
        tp = sum(1 for (_, _, _, gt, spn, _) in test_results if gt and spn)
        fp = sum(1 for (_, _, _, gt, spn, _) in test_results if not gt and spn)
        fn = sum(1 for (_, _, _, gt, spn, _) in test_results if gt and not spn)
        tn = sum(1 for (_, _, _, gt, spn, _) in test_results if not gt and not spn)

        # Overall metrics
        overall_acc = (tp + tn) / len(test_results) if test_results else 0.0
        overall_precision, overall_recall, overall_f1 = count_precision_recall_f1(
            tp, fp, fn
        )

        # Skeleton-specific metrics
        skeleton_results = [
            (gt, spn) for (_, _, z, gt, spn, _) in test_results if len(z) == 0
        ]
        if skeleton_results:
            skel_tp = sum(1 for (gt, spn) in skeleton_results if gt and spn)
            skel_fp = sum(1 for (gt, spn) in skeleton_results if not gt and spn)
            skel_fn = sum(1 for (gt, spn) in skeleton_results if gt and not spn)
            skel_tn = sum(1 for (gt, spn) in skeleton_results if not gt and not spn)
            skel_acc = (skel_tp + skel_tn) / len(skeleton_results)
            skel_precision, skel_recall, skel_f1 = count_precision_recall_f1(
                skel_tp, skel_fp, skel_fn
            )
        else:
            skel_acc = 0.0
            skel_precision, skel_recall, skel_f1 = None, None, None

        results.update(
            {
                "n_tests": len(test_cases),
                "n_skeleton_tests": len(skeleton_results),
                "n_conditional_tests": len(test_cases) - len(skeleton_results),
                "overall_accuracy": overall_acc,
                "overall_f1": overall_f1 if overall_f1 is not None else 0.0,
                "overall_precision": overall_precision
                if overall_precision is not None
                else 0.0,
                "overall_recall": overall_recall if overall_recall is not None else 0.0,
                "skeleton_accuracy": skel_acc,
                "skeleton_f1": skel_f1 if skel_f1 is not None else 0.0,
                "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
            }
        )

    except Exception as e:
        logging.warning(f"Error evaluating independence structure for {name}: {e}")

    return results


def log_independence_structure_results(results, name=None):
    """
    Log independence structure evaluation results.

    Args:
        results: Dictionary from evaluate_spn_independence_structure
        name: Optional name override
    """
    name = name or results.get("name", "SPN")

    logging.info(f"  [{name}] Independence Structure:")

    if "n_tests" in results:
        n_total = results["n_tests"]
        n_skel = results.get("n_skeleton_tests", 0)
        n_cond = results.get("n_conditional_tests", 0)
        logging.info(
            f"    Tests: {n_total} total ({n_skel} skeleton, {n_cond} conditional)"
        )

    if "overall_accuracy" in results:
        acc = results["overall_accuracy"]
        status = "✓" if acc > 0.75 else "✗"
        logging.info(f"    Overall Accuracy: {acc:.3f} {status}")

    if "overall_f1" in results:
        f1 = results["overall_f1"]
        logging.info(f"    Overall F1: {f1:.3f}")

    if "skeleton_accuracy" in results:
        skel_acc = results["skeleton_accuracy"]
        status = "✓" if skel_acc > 0.80 else "✗"
        logging.info(f"    Skeleton Accuracy: {skel_acc:.3f} {status}")

    if "confusion_matrix" in results:
        cm = results["confusion_matrix"]
        logging.info(
            f"    Confusion: TP={cm['TP']}, FP={cm['FP']}, FN={cm['FN']}, TN={cm['TN']}"
        )


def log_spn_quality(results, name=None):
    """
    Log SPN quality metrics in a formatted way.

    Args:
        results: Dictionary from evaluate_spn_quality
        name: Optional name override
    """
    name = name or results.get("name", "SPN")

    logging.info(f"  [{name}] Quality Metrics:")

    if "train_ll" in results:
        logging.info(f"    Train LL: {results['train_ll']:.4f}")

    if "mmd_pvalue" in results:
        mmd_p = results["mmd_pvalue"]
        mmd_sq = results.get("mmd_squared", None)
        status = "✓" if mmd_p > 0.05 else "✗"
        if mmd_sq is not None:
            logging.info(f"    MMD²: {mmd_sq:.6f}, p-value: {mmd_p:.3f} {status}")
        else:
            logging.info(f"    MMD p-value: {mmd_p:.3f} {status}")

    if "ks_fail_ratio" in results:
        ks_ratio = results["ks_fail_ratio"]
        status = "✓" if ks_ratio < 0.5 else "✗"
        logging.info(f"    KS test: {int(ks_ratio*100)}% failed {status}")


# From spn_umap_visualization.py - extract_spn_samples function
def extract_spn_samples(
    spn_model,
    n_samples: int = 500,
    device: str = "cpu",
    has_context: bool = True,
) -> np.ndarray:
    """
    Extract samples from an SPN model.

    Args:
        spn_model: Trained SPN (LocalSPNWrapper or GlobalFedSPN)
        n_samples: Number of samples to generate
        device: torch device
        has_context: Whether to remove context column

    Returns:
        samples: [n_samples, d] numpy array
    """
    try:
        with torch.no_grad():
            samples = spn_model.sample(n_samples).cpu().numpy()

        # Remove context column if present
        if has_context and samples.shape[1] > 1:
            samples = samples[:, :-1]

        return samples
    except Exception as e:
        logging.error(f"Failed to sample from SPN: {e}")
        return None
