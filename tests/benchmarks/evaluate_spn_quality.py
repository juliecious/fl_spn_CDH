#!/usr/bin/env python3
"""
SPN Quality Evaluation Framework

Validates that local and global SPNs correctly learn data distributions.
Two-level evaluation: Local SPNs (per client) + Global federated SPN

Metrics:
1. Log-likelihood (train/test split) - Native SPN metric
2. Maximum Mean Discrepancy (MMD²) - Distribution comparison (Gretton et al. 2012)
3. Kolmogorov-Smirnov test (per dimension) - Marginal distributions
4. UMAP visualization (d>2, supplementary only) - Qualitative comparison

Usage:
    from evaluate_spn_quality import SPNEvaluator

    evaluator = SPNEvaluator(X, c_indx, K, scenario="horizontal")
    evaluator.prepare_train_test_split(test_ratio=0.2)

    # Evaluate local SPNs
    local_results = evaluator.evaluate_local_spn(spn_k, client_id=k)

    # Evaluate global federated SPN
    global_results = evaluator.evaluate_global_spn(global_spn)

    # Generate report
    evaluator.generate_report(output_dir)

References:
- Gretton et al. 2012: "A Kernel Two-Sample Test" (JMLR)
- Poon & Domingos 2011: "Sum-product networks" (UAI)
- McInnes et al. 2018: "UMAP" (arXiv)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from umap import UMAP

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class SPNEvaluator:
    """
    Two-level SPN quality evaluation: local (per client) + global (federated).

    Evaluates density estimation quality before causal discovery.
    """

    def __init__(
        self,
        X: np.ndarray,
        c_indx: np.ndarray,
        K: int,
        scenario: str,
        device="cuda",
        include_context=True,
    ):
        """
        Initialize SPN evaluator.

        Args:
            X: Data matrix (n_total, d)
            c_indx: Domain indices (n_total, 1), values in {0, ..., K-1}
            K: Number of clients
            scenario: 'horizontal', 'vertical', or 'hybrid'
            device: 'cuda' or 'cpu'
            include_context: Whether SPNs were trained with context column (default True)
        """
        self.X = X
        self.c_indx = c_indx.flatten()
        self.K = K
        self.scenario = scenario
        self.device = device
        self.include_context = include_context

        self.n_total, self.d = X.shape

        # If SPNs trained with context, augment X with context column
        if include_context:
            self.X_aug = np.concatenate([X, c_indx], axis=1)
            self.d_aug = self.d + 1
        else:
            self.X_aug = X
            self.d_aug = self.d

        # Will be populated by prepare_train_test_split()
        self.X_train_splits = None
        self.X_test_splits = None
        self.X_train_global = None
        self.X_test_global = None

        # Results storage
        self.local_results = []
        self.global_results = {}

        print(f"[SPNEvaluator] Initialized: n={self.n_total}, d={self.d}, K={K}")
        print(f"[SPNEvaluator] Scenario: {scenario}, Device: {device}")

    def prepare_train_test_split(self, test_ratio: float = 0.2, seed: int = 42):
        """
        Stratified train/test split per client (80/20 default).

        Ensures each client retains same proportion in train/test.
        Held-out test set used for evaluation (never seen by SPN during training).

        Args:
            test_ratio: Proportion of data for test (default 0.2)
            seed: Random seed for reproducibility

        Returns:
            Sets self.X_train_splits, self.X_test_splits (lists of length K)
            Sets self.X_train_global, self.X_test_global (concatenated)
        """
        np.random.seed(seed)

        X_train_splits = []
        X_test_splits = []

        for k in range(self.K):
            # Get client k's data (use augmented if context included)
            client_mask = self.c_indx == k
            X_client = self.X_aug[client_mask]
            n_client = len(X_client)

            # Shuffle and split
            indices = np.random.permutation(n_client)
            split_idx = int((1 - test_ratio) * n_client)

            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]

            X_train_k = X_client[train_indices]
            X_test_k = X_client[test_indices]

            X_train_splits.append(X_train_k)
            X_test_splits.append(X_test_k)

            print(f"[Split] Client {k}: train={len(X_train_k)}, test={len(X_test_k)}")

        # Store splits
        self.X_train_splits = X_train_splits
        self.X_test_splits = X_test_splits

        # Global concatenation
        self.X_train_global = np.vstack(X_train_splits)
        self.X_test_global = np.vstack(X_test_splits)

        print(
            f"[Split] Global: train={len(self.X_train_global)}, test={len(self.X_test_global)}"
        )
        print(
            f"[Split] Data dimensionality: d={self.d}, d_aug={self.d_aug} (include_context={self.include_context})"
        )

    def evaluate_local_spn(
        self,
        spn,
        client_id: int,
        n_samples_mmd: int = 1000,
        n_permutations: int = 1000,
    ) -> Dict:
        """
        Evaluate local SPN quality for client k.

        Metrics:
        1. Train/test log-likelihood
        2. Overfitting gap
        3. MMD² with RBF kernel
        4. MMD p-value (permutation test)
        5. KS test per dimension

        Args:
            spn: Trained local SPN (has .log_prob() and .sample() methods)
            client_id: Client index (0 to K-1)
            n_samples_mmd: Number of samples to generate for MMD (default 1000)
            n_permutations: Number of permutations for MMD p-value (default 1000)

        Returns:
            results: Dict with metrics
        """
        if self.X_train_splits is None:
            raise ValueError("Must call prepare_train_test_split() first")

        X_train = self.X_train_splits[client_id]
        X_test = self.X_test_splits[client_id]

        print(f"\n[Local SPN] Evaluating Client {client_id}")

        # Convert to torch tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        # 1. Log-likelihood
        with torch.no_grad():
            train_ll_t = spn.log_prob(X_train_t).cpu().numpy()
            test_ll_t = spn.log_prob(X_test_t).cpu().numpy()

        train_ll = float(np.mean(train_ll_t))
        test_ll = float(np.mean(test_ll_t))
        overfitting_gap = abs(train_ll - test_ll) / abs(train_ll)

        print(f"  Train LL: {train_ll:.3f}")
        print(f"  Test LL:  {test_ll:.3f}")
        print(f"  Overfitting gap: {overfitting_gap:.3f} (threshold: <0.20)")

        # 2. Generate samples for MMD
        n_samples = min(n_samples_mmd, len(X_test))
        with torch.no_grad():
            X_gen_t = spn.sample(n_samples)
            if X_gen_t.dim() == 3:  # [n, 1, d]
                X_gen_t = X_gen_t.squeeze(1)
            X_gen = X_gen_t.cpu().numpy()

        # Ensure correct shape and handle context column
        if X_gen.ndim == 1:
            X_gen = X_gen.reshape(-1, 1)

        # If generated samples include context column, exclude it for comparison
        if self.include_context and X_gen.shape[1] == self.d_aug:
            X_gen = X_gen[:, : self.d]  # Keep only feature columns, drop context
            X_test_subset = X_test[
                :n_samples, : self.d
            ]  # Also drop context from test data
            print(
                f"  Note: Comparing feature space only (d={self.d}, excluding context column)"
            )
        elif X_gen.shape[1] != self.d:
            print(
                f"  WARNING: Generated samples have shape {X_gen.shape}, expected (*, {self.d})"
            )
            X_test_subset = X_test[:n_samples]
        else:
            X_test_subset = X_test[:n_samples]

        # 3. MMD² with RBF kernel
        mmd_squared = self._compute_mmd_squared(X_test_subset, X_gen)
        print(f"  MMD² (×10⁻³): {mmd_squared * 1000:.2f}")

        # 4. MMD p-value (permutation test)
        mmd_pvalue = self._compute_mmd_pvalue(
            X_test_subset, X_gen, mmd_squared, n_permutations=n_permutations
        )
        print(f"  MMD p-value: {mmd_pvalue:.3f} (threshold: >0.05)")

        # 5. KS test per dimension
        ks_results = self._compute_ks_per_dimension(X_test_subset, X_gen)
        print(
            f"  KS min p-value: {ks_results['min_pvalue']:.3f} (Bonferroni: >{0.05/self.d:.3f})"
        )
        print(
            f"  KS failed dims: {ks_results['num_failed']}/{self.d} (threshold: <30%)"
        )

        # Compile results
        results = {
            "client_id": client_id,
            "train_ll": train_ll,
            "test_ll": test_ll,
            "overfitting_gap": overfitting_gap,
            "mmd_squared": mmd_squared,
            "mmd_pvalue": mmd_pvalue,
            "ks_min_pvalue": ks_results["min_pvalue"],
            "ks_failed_dims": ks_results["num_failed"],
            "ks_pvalues": ks_results["pvalues"],
            "generated_samples": X_gen,  # Store for UMAP visualization
        }

        self.local_results.append(results)
        return results

    def evaluate_global_spn(
        self,
        global_spn,
        n_samples_mmd: int = 2000,
        n_permutations: int = 1000,
        weights: Optional[np.ndarray] = None,
        ll_before_em: Optional[float] = None,
        ll_after_em: Optional[float] = None,
    ) -> Dict:
        """
        Evaluate global federated SPN quality.

        Metrics:
        1. Global test log-likelihood
        2. MMD² (global test vs generated)
        3. Aggregation validation (scenario-specific)

        Args:
            global_spn: Trained federated SPN (has .log_prob() and .sample())
            n_samples_mmd: Number of samples for MMD
            n_permutations: Number of permutations for MMD p-value
            weights: Mixture weights (for horizontal/hybrid), optional
            ll_before_em: LL before EM refinement (if applicable)
            ll_after_em: LL after EM refinement (if applicable)

        Returns:
            results: Dict with metrics
        """
        if self.X_test_global is None:
            raise ValueError("Must call prepare_train_test_split() first")

        print(f"\n[Global SPN] Evaluating Federated Model")

        X_test = self.X_test_global

        # Convert to torch
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        # 1. Global test log-likelihood
        with torch.no_grad():
            test_ll_t = global_spn.log_prob(X_test_t).cpu().numpy()

        test_ll = float(np.mean(test_ll_t))
        print(f"  Global Test LL: {test_ll:.3f}")

        # Compare to weighted average of local test LLs
        if len(self.local_results) > 0:
            local_test_lls = [r["test_ll"] for r in self.local_results]
            local_test_sizes = [len(self.X_test_splits[k]) for k in range(self.K)]
            weighted_avg_ll = np.average(local_test_lls, weights=local_test_sizes)
            print(f"  Weighted avg of local test LLs: {weighted_avg_ll:.3f}")
            print(
                f"  Difference: {test_ll - weighted_avg_ll:.3f} (expect ~0 for good aggregation)"
            )

        # 2. Generate samples for MMD
        n_samples = min(n_samples_mmd, len(X_test))
        with torch.no_grad():
            X_gen_t = global_spn.sample(n_samples)
            if X_gen_t.dim() == 3:
                X_gen_t = X_gen_t.squeeze(1)
            X_gen = X_gen_t.cpu().numpy()

        if X_gen.ndim == 1:
            X_gen = X_gen.reshape(-1, 1)

        # If generated samples include context column, exclude it for comparison
        if self.include_context and X_gen.shape[1] == self.d_aug:
            X_gen = X_gen[:, : self.d]  # Keep only feature columns, drop context
            X_test_subset = X_test[
                :n_samples, : self.d
            ]  # Also drop context from test data
            print(
                f"  Note: Comparing feature space only (d={self.d}, excluding context column)"
            )
        else:
            X_test_subset = X_test[:n_samples]

        # 3. MMD²
        mmd_squared = self._compute_mmd_squared(X_test_subset, X_gen)
        print(f"  MMD² (×10⁻³): {mmd_squared * 1000:.2f}")

        # 4. MMD p-value
        mmd_pvalue = self._compute_mmd_pvalue(
            X_test_subset, X_gen, mmd_squared, n_permutations=n_permutations
        )
        print(f"  MMD p-value: {mmd_pvalue:.3f} (threshold: >0.05)")

        # 5. Aggregation-specific checks
        aggregation_checks = {}

        if self.scenario == "horizontal" or self.scenario == "hybrid":
            if weights is not None:
                weight_sum = float(np.sum(weights))
                weights_valid = np.all(weights >= 0)
                aggregation_checks["weight_sum"] = weight_sum
                aggregation_checks["weights_nonnegative"] = weights_valid
                print(f"  Weight sum: {weight_sum:.6f} (expect 1.0)")
                print(f"  Weights non-negative: {weights_valid}")

                # Check if sample-proportional
                expected_weights = np.array(
                    [len(self.X_train_splits[k]) for k in range(self.K)]
                )
                expected_weights = expected_weights / expected_weights.sum()
                weight_diff = np.abs(weights - expected_weights).max()
                print(
                    f"  Max weight deviation from sample-proportional: {weight_diff:.3f}"
                )
                aggregation_checks["weight_deviation"] = float(weight_diff)

        # EM convergence check
        if ll_before_em is not None and ll_after_em is not None:
            em_improved = ll_after_em >= ll_before_em
            em_gain = ll_after_em - ll_before_em
            aggregation_checks["em_improved"] = em_improved
            aggregation_checks["em_ll_gain"] = float(em_gain)
            print(f"  EM improved LL: {em_improved} (gain: {em_gain:.3f})")

        # Compile results
        results = {
            "test_ll": test_ll,
            "mmd_squared": mmd_squared,
            "mmd_pvalue": mmd_pvalue,
            "aggregation_checks": aggregation_checks,
            "generated_samples": X_gen,  # Store for UMAP visualization
        }

        self.global_results = results
        return results

    def _compute_mmd_squared(
        self, X: np.ndarray, Y: np.ndarray, kernel: str = "rbf"
    ) -> float:
        """
        Compute MMD² using RBF kernel with median bandwidth heuristic.

        Unbiased estimator from Gretton et al. 2012.

        Args:
            X: Real data (m, d)
            Y: Generated data (n, d)
            kernel: 'rbf' (Gaussian kernel)

        Returns:
            mmd_squared: MMD² value (scalar)
        """
        m = len(X)
        n = len(Y)

        if m < 2 or n < 2:
            print("  WARNING: Need at least 2 samples per set for MMD")
            return 0.0

        # Compute median bandwidth (Gretton et al. 2012 heuristic)
        # Use subset for efficiency if large
        subset_size = min(500, m)
        X_subset = X[np.random.choice(m, subset_size, replace=False)]
        pairwise_dists = pairwise_distances(X_subset, X_subset)
        median_dist = np.median(pairwise_dists[pairwise_dists > 0])
        bandwidth = median_dist

        if bandwidth == 0:
            bandwidth = 1.0  # Fallback

        # RBF kernel: k(x,y) = exp(-||x-y||²/(2σ²))
        def rbf_kernel(A, B):
            pairwise = pairwise_distances(A, B, metric="euclidean")
            return np.exp(-(pairwise**2) / (2 * bandwidth**2))

        # Unbiased MMD² estimator
        # MMD² = 1/(m(m-1)) Σ_{i≠j} k(x_i, x_j) + 1/(n(n-1)) Σ_{i≠j} k(y_i, y_j)
        #        - 2/(mn) Σ_i Σ_j k(x_i, y_j)

        K_XX = rbf_kernel(X, X)
        K_YY = rbf_kernel(Y, Y)
        K_XY = rbf_kernel(X, Y)

        # Remove diagonal for unbiased estimator
        np.fill_diagonal(K_XX, 0)
        np.fill_diagonal(K_YY, 0)

        term1 = K_XX.sum() / (m * (m - 1))
        term2 = K_YY.sum() / (n * (n - 1))
        term3 = K_XY.sum() / (m * n)

        mmd_squared = term1 + term2 - 2 * term3

        return float(mmd_squared)

    def _compute_mmd_pvalue(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        mmd_obs: float,
        n_permutations: int = 1000,
    ) -> float:
        """
        Compute p-value for MMD via permutation test.

        H0: X and Y come from same distribution
        p-value = (1 + #{MMD_perm >= MMD_obs}) / (1 + n_permutations)

        Args:
            X: Real data (m, d)
            Y: Generated data (n, d)
            mmd_obs: Observed MMD² value
            n_permutations: Number of permutations

        Returns:
            p_value: Permutation test p-value
        """
        m = len(X)
        n = len(Y)
        combined = np.vstack([X, Y])

        count_greater = 0

        for _ in range(n_permutations):
            # Permute combined data
            perm_idx = np.random.permutation(m + n)
            X_perm = combined[perm_idx[:m]]
            Y_perm = combined[perm_idx[m:]]

            # Compute MMD² on permuted data
            mmd_perm = self._compute_mmd_squared(X_perm, Y_perm)

            if mmd_perm >= mmd_obs:
                count_greater += 1

        p_value = (1 + count_greater) / (1 + n_permutations)
        return float(p_value)

    def _compute_ks_per_dimension(
        self, X: np.ndarray, Y: np.ndarray, alpha: float = 0.05
    ) -> Dict:
        """
        Kolmogorov-Smirnov test for each dimension independently.

        Tests marginal distributions per dimension.
        Uses Bonferroni correction: α' = α / d

        Args:
            X: Real data (m, d)
            Y: Generated data (n, d)
            alpha: Significance level (default 0.05)

        Returns:
            results: Dict with min_pvalue, num_failed, pvalues
        """
        d = X.shape[1]
        pvalues = []

        bonferroni_alpha = alpha / d

        for j in range(d):
            stat, pval = ks_2samp(X[:, j], Y[:, j])
            pvalues.append(pval)

        min_pvalue = min(pvalues)
        num_failed = sum(1 for p in pvalues if p < bonferroni_alpha)

        return {
            "min_pvalue": min_pvalue,
            "num_failed": num_failed,
            "pvalues": pvalues,
        }

    def visualize_umap(
        self,
        X_real: np.ndarray,
        X_gen: np.ndarray,
        title: str = "UMAP: Real vs Generated",
        save_path: Optional[Path] = None,
    ):
        """
        UMAP visualization for multivariate data (d>2).

        Supplementary qualitative visualization only - not a rigorous metric.
        Should only be used alongside quantitative metrics (MMD, KS).

        Args:
            X_real: Real data (n, d)
            X_gen: Generated data (n, d)
            title: Plot title
            save_path: Path to save figure (optional)

        Returns:
            fig: Matplotlib figure object
        """
        if not UMAP_AVAILABLE:
            print("  [UMAP] Warning: umap-learn not installed, skipping visualization")
            print("  Install with: pip install umap-learn")
            return None

        if self.d <= 2:
            print(
                f"  [UMAP] Warning: d={self.d} <= 2, UMAP not needed. Use scatter plot instead."
            )
            return None

        print(f"\n[UMAP] Generating visualization for d={self.d}")

        # Combine data for UMAP
        X_combined = np.vstack([X_real, X_gen])
        labels = np.array(["Real"] * len(X_real) + ["Generated"] * len(X_gen))

        # Fit UMAP (dimensionality reduction to 2D)
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(X_combined)

        # Split back
        X_real_umap = X_umap[: len(X_real)]
        X_gen_umap = X_umap[len(X_real) :]

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot real (blue) vs generated (orange)
        ax.scatter(
            X_real_umap[:, 0],
            X_real_umap[:, 1],
            c="steelblue",
            alpha=0.5,
            s=20,
            label="Real",
            edgecolors="none",
        )
        ax.scatter(
            X_gen_umap[:, 0],
            X_gen_umap[:, 1],
            c="coral",
            alpha=0.5,
            s=20,
            label="Generated",
            edgecolors="none",
        )

        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved: {save_path}")

        return fig

    def generate_report(self, output_dir: Path, include_umap: bool = True):
        """
        Generate evaluation report with tables and figures.

        Creates:
        - table1_local_evaluation.csv
        - table2_global_evaluation.csv
        - evaluation_summary.txt
        - (optional) figure1_umap_global.png (if include_umap=True and d>2)
        - (optional) figure2_umap_local_k{i}.png (per client, if include_umap=True)

        Args:
            output_dir: Directory to save reports
            include_umap: Generate UMAP visualizations (default True)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Report] Generating evaluation report in {output_dir}")

        # Table 1: Local SPN evaluation
        if len(self.local_results) > 0:
            df_local = pd.DataFrame(
                [
                    {
                        "client": r["client_id"],
                        "train_ll": r["train_ll"],
                        "test_ll": r["test_ll"],
                        "overfitting_gap": r["overfitting_gap"],
                        "mmd_squared_e3": r["mmd_squared"] * 1000,
                        "mmd_pvalue": r["mmd_pvalue"],
                        "ks_min_pvalue": r["ks_min_pvalue"],
                        "ks_failed_dims": r["ks_failed_dims"],
                    }
                    for r in self.local_results
                ]
            )

            # Add mean row
            mean_row = {
                "client": "Mean",
                "train_ll": df_local["train_ll"].mean(),
                "test_ll": df_local["test_ll"].mean(),
                "overfitting_gap": df_local["overfitting_gap"].mean(),
                "mmd_squared_e3": df_local["mmd_squared_e3"].mean(),
                "mmd_pvalue": df_local["mmd_pvalue"].mean(),
                "ks_min_pvalue": df_local["ks_min_pvalue"].mean(),
                "ks_failed_dims": df_local["ks_failed_dims"].mean(),
            }
            df_local = pd.concat(
                [df_local, pd.DataFrame([mean_row])], ignore_index=True
            )

            csv_path = output_dir / "table1_local_evaluation.csv"
            df_local.to_csv(csv_path, index=False, float_format="%.4f")
            print(f"  Saved: {csv_path}")

        # Table 2: Global evaluation
        if self.global_results:
            global_data = {
                "model": "Federated",
                "test_ll": self.global_results["test_ll"],
                "mmd_squared_e3": self.global_results["mmd_squared"] * 1000,
                "mmd_pvalue": self.global_results["mmd_pvalue"],
            }

            # Add aggregation checks
            agg = self.global_results.get("aggregation_checks", {})
            if "weight_sum" in agg:
                global_data["weight_sum"] = agg["weight_sum"]
            if "em_improved" in agg:
                global_data["em_ll_gain"] = agg.get("em_ll_gain", 0)

            df_global = pd.DataFrame([global_data])
            csv_path = output_dir / "table2_global_evaluation.csv"
            df_global.to_csv(csv_path, index=False, float_format="%.4f")
            print(f"  Saved: {csv_path}")

        # Summary text
        summary_path = output_dir / "evaluation_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("SPN Quality Evaluation Summary\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Dataset: n={self.n_total}, d={self.d}, K={self.K}\n")
            f.write(f"Scenario: {self.scenario}\n\n")

            f.write("Thresholds:\n")
            f.write("  - Overfitting gap: <0.20 (acceptable)\n")
            f.write("  - MMD p-value: >0.05 (fail to reject same distribution)\n")
            f.write(f"  - KS min p-value: >{0.05/self.d:.3f} (Bonferroni)\n")
            f.write("  - KS failed dims: <30% of dimensions\n\n")

            if len(self.local_results) > 0:
                f.write("Local SPN Results:\n")
                for r in self.local_results:
                    f.write(f"  Client {r['client_id']}:\n")
                    f.write(f"    Train LL: {r['train_ll']:.3f}\n")
                    f.write(f"    Test LL: {r['test_ll']:.3f}\n")
                    f.write(f"    Overfitting gap: {r['overfitting_gap']:.3f}\n")
                    f.write(
                        f"    MMD² (×10⁻³): {r['mmd_squared'] * 1000:.2f}, p={r['mmd_pvalue']:.3f}\n"
                    )
                    f.write(f"    KS failed: {r['ks_failed_dims']}/{self.d} dims\n\n")

            if self.global_results:
                f.write("Global Federated SPN Results:\n")
                f.write(f"  Test LL: {self.global_results['test_ll']:.3f}\n")
                f.write(
                    f"  MMD² (×10⁻³): {self.global_results['mmd_squared'] * 1000:.2f}, "
                )
                f.write(f"p={self.global_results['mmd_pvalue']:.3f}\n")

        print(f"  Saved: {summary_path}")

        # Generate UMAP visualizations (optional)
        if include_umap and UMAP_AVAILABLE and self.d > 2:
            print("\n[UMAP] Generating visualizations...")

            # Global UMAP
            if self.global_results and "generated_samples" in self.global_results:
                X_gen_global = self.global_results["generated_samples"]
                fig_global = self.visualize_umap(
                    self.X_test_global,
                    X_gen_global,
                    title=f"UMAP: Global Federated SPN (d={self.d}, K={self.K})",
                    save_path=output_dir / "figure1_umap_global.png",
                )
                if fig_global:
                    plt.close(fig_global)

            # Local UMAPs
            for i, result in enumerate(self.local_results):
                if "generated_samples" in result:
                    X_gen_local = result["generated_samples"]
                    fig_local = self.visualize_umap(
                        self.X_test_splits[result["client_id"]],
                        X_gen_local,
                        title=f"UMAP: Local SPN Client {result['client_id']} (d={self.d})",
                        save_path=output_dir
                        / f"figure2_umap_local_k{result['client_id']}.png",
                    )
                    if fig_local:
                        plt.close(fig_local)

        elif include_umap and not UMAP_AVAILABLE:
            print("\n[UMAP] Skipped: umap-learn not installed")
            print("  Install with: pip install umap-learn")

        print("\n[Report] Evaluation complete!")


# Convenience function for standalone usage
def evaluate_fedcdh_spns(
    fedcdh_model, X, c_indx, K, scenario, output_dir, test_ratio=0.2
):
    """
    Convenience function to evaluate SPNs from trained FedCDH model.

    Args:
        fedcdh_model: Trained FedCDH instance
        X: Data matrix (n_total, d)
        c_indx: Domain indices (n_total, 1)
        K: Number of clients
        scenario: 'horizontal', 'vertical', or 'hybrid'
        output_dir: Output directory for reports
        test_ratio: Test set proportion (default 0.2)

    Returns:
        evaluator: SPNEvaluator instance with results
    """
    # TODO: Extract local and global SPNs from fedcdh_model
    # This requires exposing these in FedCDH.py
    raise NotImplementedError(
        "Need to expose local_spns and fed_spn_model from FedCDH class"
    )


if __name__ == "__main__":
    print("SPN Quality Evaluation Framework")
    print("Import this module and use SPNEvaluator class")
    print("\nExample usage:")
    print("  evaluator = SPNEvaluator(X, c_indx, K, scenario='horizontal')")
    print("  evaluator.prepare_train_test_split()")
    print("  results = evaluator.evaluate_local_spn(spn, client_id=0)")
    print("  evaluator.generate_report('output/')")
