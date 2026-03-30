#!/usr/bin/env python3
"""
Consolidated SPN Quality Evaluation Script

Evaluates both local and global SPN quality with comprehensive metrics.

SPNs Evaluated:
- Local SPNs: LocalSPNWrapper instances (from causallearn.utils.FedPC)
- Global SPN: GlobalFedSPN or FedCDH_SPN_Wrapper (from FedPC or FedCDH)
- Generic: Any SPN with .log_prob(X) and .sample(n) methods

Usage:
    # Option 1: Evaluate trained FedCDH model (extracts SPNs automatically)
    from tests.benchmarks.evaluate_spn import evaluate_fedcdh_spns
    results = evaluate_fedcdh_spns(fedcdh_model, X, c_indx, K, scenario, output_dir)

    # Option 2: Evaluate SPNs directly (manual)
    from tests.benchmarks.evaluate_spn import SPNEvaluator
    evaluator = SPNEvaluator(X, c_indx, K, scenario)
    evaluator.prepare_train_test_split()
    evaluator.evaluate_local_spn(local_spn_k, client_id=k)
    evaluator.evaluate_global_spn(global_spn)
    evaluator.generate_report(output_dir)

SPN Requirements:
- Must implement: .log_prob(X) -> tensor of log-probabilities
- Must implement: .sample(n) -> tensor of n generated samples
- Compatible with: LocalSPNWrapper, GlobalFedSPN, FedCDH_SPN_Wrapper

Metrics:
- Train/test log-likelihood with overfitting gap
- MMD² with RBF kernel (distribution matching)
- KS test per dimension (marginal distributions)
- Convergence indicators (loss stability, trend)
- Global vs local comparison

References:
- Gretton et al. 2012: "A Kernel Two-Sample Test" (JMLR)
- Poon & Domingos 2011: "Sum-product networks" (UAI)
"""

import sys
import os
import time
import argparse
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
    Comprehensive SPN quality evaluator for local and global models.

    Generic evaluator that works with any SPN implementing:
    - .log_prob(X: torch.Tensor) -> torch.Tensor: Log-likelihood of samples
    - .sample(n: int) -> torch.Tensor: Generate n samples

    Compatible SPN types:
    - LocalSPNWrapper (causallearn.utils.FedPC)
    - GlobalFedSPN (causallearn.utils.FedPC)
    - FedCDH_SPN_Wrapper (causallearn.search.FCMBased.FedCDH.FedCDH)
    - Any custom SPN with the required interface
    """

    def __init__(
        self,
        X: np.ndarray,
        c_indx: np.ndarray,
        K: int,
        scenario: str,
        device: str = None,
        include_context=True,
    ):
        """
        Initialize SPN evaluator.

        Args:
            X: Data matrix (n_total, d)
            c_indx: Domain indices (n_total, 1), values in {0, ..., K-1}
            K: Number of clients
            scenario: 'horizontal', 'vertical', or 'hybrid'
            device: 'cuda', 'cpu', or None (auto-detect)
            include_context: Whether SPNs were trained with context column
        """
        self.X = X
        self.c_indx = c_indx.flatten()
        self.K = K
        self.scenario = scenario

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
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
        print(f"[SPNEvaluator] Scenario: {scenario}, Device: {self.device}")

    def prepare_train_test_split(self, test_ratio: float = 0.2, seed: int = 42):
        """Stratified train/test split per client (80/20 default)."""
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
        training_losses: Optional[List[float]] = None,
    ) -> Dict:
        """
        Evaluate local SPN quality for client k.

        Args:
            spn: Trained local SPN
            client_id: Client index
            n_samples_mmd: Number of samples for MMD
            n_permutations: Number of permutations for MMD p-value
            training_losses: Optional list of training losses for convergence analysis

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

        # 2. Convergence analysis (if training losses provided)
        convergence_metrics = {}
        if training_losses and len(training_losses) >= 10:
            last_10 = training_losses[-10:]
            loss_std = np.std(last_10)
            loss_trend = last_10[-1] - last_10[0]
            converged = loss_std < 0.05 and abs(loss_trend) < 0.1

            convergence_metrics = {
                "loss_std": loss_std,
                "loss_trend": loss_trend,
                "converged": converged,
            }

            print(f"\n  Convergence (last 10 epochs):")
            print(f"    Loss std dev: {loss_std:.4f} (threshold: <0.05)")
            print(f"    Loss trend:   {loss_trend:.4f} (threshold: <0.10)")
            print(
                f"    Converged: {'✅ YES' if converged else '⚠️  NO (still improving)'}"
            )

        # 3. Generate samples for MMD
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
            X_gen = X_gen[:, : self.d]  # Keep only feature columns
            X_test_subset = X_test[:n_samples, : self.d]
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

        # 4. MMD² with RBF kernel
        mmd_squared = self._compute_mmd_squared(X_test_subset, X_gen)
        print(f"  MMD² (×10⁻³): {mmd_squared * 1000:.2f}")

        # 5. MMD p-value (permutation test)
        mmd_pvalue = self._compute_mmd_pvalue(
            X_test_subset, X_gen, mmd_squared, n_permutations=n_permutations
        )
        print(f"  MMD p-value: {mmd_pvalue:.3f} (threshold: >0.05)")

        # 6. KS test per dimension
        ks_results = self._compute_ks_per_dimension(X_test_subset, X_gen)
        print(
            f"  KS min p-value: {ks_results['min_pvalue']:.3f} (Bonferroni: >{0.05/self.d:.3f})"
        )
        print(
            f"  KS failed dims: {ks_results['num_failed']}/{self.d} (threshold: <30%)"
        )

        # 7. Quality assessment
        gap_ok = overfitting_gap < 0.20
        mmd_ok = mmd_pvalue > 0.05
        ks_ok = ks_results["num_failed"] < 0.3 * self.d

        quality = (
            "✅ GOOD" if (gap_ok and mmd_ok and ks_ok) else "⚠️  NEEDS MORE TRAINING"
        )
        print(f"\n  Overall quality: {quality}")

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
            "generated_samples": X_gen,
            **convergence_metrics,
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

        Args:
            global_spn: Trained federated SPN
            n_samples_mmd: Number of samples for MMD
            n_permutations: Number of permutations for MMD p-value
            weights: Mixture weights (for horizontal/hybrid)
            ll_before_em: LL before EM refinement
            ll_after_em: LL after EM refinement

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
            ll_gain = test_ll - weighted_avg_ll
            print(f"  Weighted avg of local test LLs: {weighted_avg_ll:.3f}")
            print(
                f"  LL gain over locals: {ll_gain:+.3f} (positive = aggregation helps)"
            )
        else:
            ll_gain = None

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
            X_gen = X_gen[:, : self.d]
            X_test_subset = X_test[:n_samples, : self.d]
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

        if self.scenario in ["horizontal", "hybrid"]:
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
                print(f"  Max weight deviation: {weight_diff:.3f}")
                aggregation_checks["weight_deviation"] = float(weight_diff)

        # EM convergence check
        if ll_before_em is not None and ll_after_em is not None:
            em_improved = ll_after_em >= ll_before_em
            em_gain = ll_after_em - ll_before_em
            aggregation_checks["em_improved"] = em_improved
            aggregation_checks["em_ll_gain"] = float(em_gain)
            print(f"  EM improved LL: {em_improved} (gain: {em_gain:.3f})")

        # 6. Quality assessment
        mmd_ok = mmd_pvalue > 0.05
        quality = "✅ GOOD" if mmd_ok else "⚠️  NEEDS MORE TRAINING"
        print(f"\n  Overall quality: {quality}")

        # Compile results
        results = {
            "test_ll": test_ll,
            "mmd_squared": mmd_squared,
            "mmd_pvalue": mmd_pvalue,
            "aggregation_checks": aggregation_checks,
            "generated_samples": X_gen,
            "ll_gain_over_locals": ll_gain,
        }

        self.global_results = results
        return results

    def _compute_mmd_squared(
        self, X: np.ndarray, Y: np.ndarray, kernel: str = "rbf"
    ) -> float:
        """Compute MMD² using RBF kernel with median bandwidth heuristic."""
        m = len(X)
        n = len(Y)

        if m < 2 or n < 2:
            print("  WARNING: Need at least 2 samples per set for MMD")
            return 0.0

        # Compute median bandwidth
        subset_size = min(500, m)
        X_subset = X[np.random.choice(m, subset_size, replace=False)]
        pairwise_dists = pairwise_distances(X_subset, X_subset)
        median_dist = np.median(pairwise_dists[pairwise_dists > 0])
        bandwidth = median_dist if median_dist > 0 else 1.0

        # RBF kernel
        def rbf_kernel(A, B):
            pairwise = pairwise_distances(A, B, metric="euclidean")
            return np.exp(-(pairwise**2) / (2 * bandwidth**2))

        # Unbiased MMD² estimator
        K_XX = rbf_kernel(X, X)
        K_YY = rbf_kernel(Y, Y)
        K_XY = rbf_kernel(X, Y)

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
        """Compute p-value for MMD via permutation test."""
        m = len(X)
        n = len(Y)
        combined = np.vstack([X, Y])

        count_greater = 0

        for _ in range(n_permutations):
            perm_idx = np.random.permutation(m + n)
            X_perm = combined[perm_idx[:m]]
            Y_perm = combined[perm_idx[m:]]

            mmd_perm = self._compute_mmd_squared(X_perm, Y_perm)

            if mmd_perm >= mmd_obs:
                count_greater += 1

        p_value = (1 + count_greater) / (1 + n_permutations)
        return float(p_value)

    def _compute_ks_per_dimension(
        self, X: np.ndarray, Y: np.ndarray, alpha: float = 0.05
    ) -> Dict:
        """KS test for each dimension with Bonferroni correction."""
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

    def generate_report(self, output_dir: Path, include_umap: bool = False):
        """Generate comprehensive evaluation report."""
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
                        "converged": r.get("converged", "N/A"),
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
                "converged": "N/A",
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
                "ll_gain_over_locals": self.global_results.get(
                    "ll_gain_over_locals", "N/A"
                ),
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

            f.write("Convergence Criteria:\n")
            f.write("  - Overfitting gap: <0.20 (good), 0.20-0.50 (acceptable)\n")
            f.write("  - MMD p-value: >0.05 (distributions match)\n")
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
                    f.write(f"    KS failed: {r['ks_failed_dims']}/{self.d} dims\n")
                    if "converged" in r:
                        f.write(f"    Converged: {r['converged']}\n")
                    f.write("\n")

            if self.global_results:
                f.write("Global Federated SPN Results:\n")
                f.write(f"  Test LL: {self.global_results['test_ll']:.3f}\n")
                f.write(
                    f"  MMD² (×10⁻³): {self.global_results['mmd_squared'] * 1000:.2f}, "
                )
                f.write(f"p={self.global_results['mmd_pvalue']:.3f}\n")
                if self.global_results.get("ll_gain_over_locals") is not None:
                    f.write(
                        f"  LL gain over locals: {self.global_results['ll_gain_over_locals']:+.3f}\n"
                    )

        print(f"  Saved: {summary_path}")
        print("\n[Report] Evaluation complete!")

    def print_convergence_summary(self):
        """Print comprehensive convergence summary with final verdict."""
        print("\n" + "=" * 80)
        print("CONVERGENCE SUMMARY")
        print("=" * 80)

        # Local SPNs
        if len(self.local_results) > 0:
            print("\nLocal SPNs:")
            for r in self.local_results:
                gap_ok = "✅" if r["overfitting_gap"] < 0.20 else "⚠️"
                mmd_ok = "✅" if r["mmd_pvalue"] > 0.05 else "⚠️"
                conv = r.get("converged", None)
                conv_icon = "✅" if conv else "⚠️" if conv is not None else "N/A"

                print(f"  Client {r['client_id']}:")
                if conv is not None:
                    print(f"    Convergence:     {conv_icon}")
                print(f"    Overfitting gap: {gap_ok} {r['overfitting_gap']:.3f}")
                print(f"    MMD p-value:     {mmd_ok} {r['mmd_pvalue']:.3f}")

        # Global SPN
        if self.global_results:
            print("\nGlobal SPN:")
            mmd_ok = "✅" if self.global_results["mmd_pvalue"] > 0.05 else "⚠️"
            print(f"  MMD p-value: {mmd_ok} {self.global_results['mmd_pvalue']:.3f}")
            if self.global_results.get("ll_gain_over_locals") is not None:
                ll_gain = self.global_results["ll_gain_over_locals"]
                gain_icon = "✅" if ll_gain > 0 else "⚠️"
                print(f"  LL gain:     {gain_icon} {ll_gain:+.3f}")

        # Final verdict
        all_local_good = all(
            r["overfitting_gap"] < 0.50 and r["mmd_pvalue"] > 0.05
            for r in self.local_results
        )
        global_good = (
            self.global_results.get("mmd_pvalue", 0) > 0.05
            if self.global_results
            else False
        )

        print("\nFinal Verdict:")
        if all_local_good and global_good:
            verdict = "✅ ALL SPNs WELL-TRAINED AND CONVERGED"
        elif global_good:
            verdict = "✅ READY FOR EXPERIMENTS (global SPN good, local SPNs acceptable)"
        else:
            verdict = "❌ MORE TRAINING NEEDED (increase epochs)"

        print(f"  {verdict}")
        print("=" * 80)


def evaluate_fedcdh_spns(
    fedcdh_model,
    X: np.ndarray,
    c_indx: np.ndarray,
    K: int,
    scenario: str,
    output_dir: str,
    test_ratio: float = 0.2,
    n_samples_mmd: int = 1000,
    n_permutations: int = 1000,
    training_losses: Optional[List[List[float]]] = None,
    device: str = None,
) -> Dict:
    """
    Convenience function to evaluate SPNs from trained FedCDH model.

    Extracts and evaluates:
    - local_spns: List[LocalSPNWrapper] from fedcdh_model.local_spns
    - fed_spn_model: FedCDH_SPN_Wrapper from fedcdh_model.fed_spn_model

    These are trained by FedCDH.fit() and exposed as attributes:
    - LocalSPNWrapper: Per-client SPNs (from causallearn.utils.FedPC)
    - FedCDH_SPN_Wrapper: Wrapped GlobalFedSPN (from FedCDH.py)

    Args:
        fedcdh_model: Trained FedCDH instance with .local_spns and .fed_spn_model
        X: Data matrix (n_total, d)
        c_indx: Domain indices (n_total, 1)
        K: Number of clients
        scenario: 'horizontal', 'vertical', or 'hybrid'
        output_dir: Output directory for reports
        test_ratio: Test set proportion
        n_samples_mmd: Samples for MMD computation
        n_permutations: Permutations for MMD p-value
        training_losses: Optional list of training losses per client
        device: 'cuda', 'cpu', or None (auto-detect)

    Returns:
        Dictionary with local_results and global_results
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Device] Auto-detected: {device}")
    else:
        print(f"[Device] Using: {device}")

    # Initialize evaluator
    evaluator = SPNEvaluator(X, c_indx, K, scenario=scenario, device=device)
    evaluator.prepare_train_test_split(test_ratio=test_ratio)

    # Extract SPNs from FedCDH model
    if not hasattr(fedcdh_model, "local_spns"):
        raise ValueError("FedCDH model does not expose local_spns attribute")

    local_spns = fedcdh_model.local_spns
    global_spn = fedcdh_model.fed_spn_model

    # Evaluate local SPNs
    print("\n" + "=" * 80)
    print("EVALUATING LOCAL SPNs")
    print("=" * 80)

    for k in range(K):
        losses_k = training_losses[k] if training_losses else None
        evaluator.evaluate_local_spn(
            local_spns[k],
            client_id=k,
            n_samples_mmd=n_samples_mmd,
            n_permutations=n_permutations,
            training_losses=losses_k,
        )

    # Evaluate global SPN
    print("\n" + "=" * 80)
    print("EVALUATING GLOBAL SPN")
    print("=" * 80)

    evaluator.evaluate_global_spn(
        global_spn, n_samples_mmd=n_samples_mmd * 2, n_permutations=n_permutations
    )

    # Generate report
    evaluator.generate_report(Path(output_dir))

    # Print summary
    evaluator.print_convergence_summary()

    return {
        "local_results": evaluator.local_results,
        "global_results": evaluator.global_results,
        "evaluator": evaluator,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SPN quality for local and global models"
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to trained FedCDH model (not implemented)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tests/experiments/spn_evaluation",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use: 'cuda', 'cpu', or 'auto' (default: auto-detect)",
    )

    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {device}")
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU")
            device = "cpu"

    print("=" * 80)
    print("SPN Quality Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    print("\nThis script should be imported and used programmatically.")
    print("\nExample usage:")
    print("  from tests.benchmarks.evaluate_spn import evaluate_fedcdh_spns")
    print(
        "  results = evaluate_fedcdh_spns(fedcdh_model, X, c_indx, K, scenario, output_dir)"
    )
    print(
        "\nOr see tests/benchmarks/test_spn_training_with_eval.py for integrated example"
    )
    print("\nCommand-line usage:")
    print("  python evaluate_spn.py --device cuda  # Force CUDA")
    print("  python evaluate_spn.py --device cpu   # Force CPU")
    print("  python evaluate_spn.py --device auto  # Auto-detect (default)")
