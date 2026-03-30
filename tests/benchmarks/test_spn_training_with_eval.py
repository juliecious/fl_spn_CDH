#!/usr/bin/env python3
"""
SPN Training with Integrated Evaluation

Monitors SPN training quality and convergence by evaluating:
1. After local SPN training (per client)
2. After global SPN aggregation and EM refinement

Provides convergence indicators to determine if SPNs are well-trained.

Usage:
    python tests/benchmarks/test_spn_training_with_eval.py
"""

import sys
import os
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.benchmarks.evaluate_spn import SPNEvaluator

print("=" * 80)
print("SPN TRAINING WITH INTEGRATED EVALUATION")
print("=" * 80)

# Configuration
class Args:
    def __init__(self):
        self.K = 2
        self.d = 6
        self.scenario = "horizontal"
        self.model_type = "linear"
        self.ci_method = "spn"
        self.n = 300  # n_per_client
        self.alpha = 0.05
        self.epochs = 100  # Increased for better convergence
        self.num_sums = 20
        self.num_leaves = 20
        self.num_repetitions = 10
        self.ablation_orientation = "mi_only"


# Generate test data
np.random.seed(42)
args = Args()
n_total = args.n * args.K
d = args.d
K = args.K

print("\nConfiguration:")
print(f"  d={d}, K={K}, n_total={n_total}, n_per_client={args.n}")
print(f"  Scenario: {args.scenario}")
print(f"  Epochs: {args.epochs}")
print(f"  SPN params: num_sums={args.num_sums}, num_leaves={args.num_leaves}")

# True DAG: chain 0->1->2->...->d-1
true_DAG = np.zeros((d, d))
for i in range(d - 1):
    true_DAG[i, i + 1] = 1

# Generate data with heterogeneity
X = np.zeros((n_total, d))
X[:, 0] = np.random.randn(n_total)

for i in range(1, d):
    domain_shift = np.repeat(np.arange(K), n_total // K) * 0.2
    X[:, i] = 0.8 * X[:, i - 1] + domain_shift + np.random.randn(n_total) * 0.5

# Normalize
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

# Domain indices
c_indx = np.repeat(np.arange(K), n_total // K).reshape(-1, 1)

# Prepare augmented data (with context column)
X_aug = np.concatenate([X, c_indx], axis=1)

# Split data
X_splits = np.array_split(X, K)

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# Initialize evaluator
print("\n" + "=" * 80)
print("INITIALIZE EVALUATOR")
print("=" * 80)

evaluator = SPNEvaluator(X, c_indx, K, scenario=args.scenario, device=device)
evaluator.prepare_train_test_split(test_ratio=0.2, seed=42)

print("\n" + "=" * 80)
print("PHASE 1: LOCAL SPN TRAINING")
print("=" * 80)

# Import SPN wrappers
from causallearn.utils.FedPC import LocalSPNWrapper

# Train local SPNs
local_spns = []
local_training_losses = []

for k in range(K):
    print(f"\n{'─' * 80}")
    print(f"Training Local SPN for Client {k}")
    print(f"{'─' * 80}")

    # Get client data (with context column)
    client_mask = c_indx.flatten() == k
    X_client = X_aug[client_mask]

    print(f"  Client {k} data shape: {X_client.shape}")
    print(f"  Training samples: {len(X_client)}")

    # Initialize local SPN
    local_d = X_client.shape[1]
    spn = LocalSPNWrapper(
        num_features=local_d,
        device=device,
        num_sums=args.num_sums,
        num_leaves=args.num_leaves,
        depth=max(1, int(np.floor(np.log2(local_d)))),
        num_repetitions=args.num_repetitions,
        seed=42 + k,
    )

    # Train with loss tracking
    print(f"\n  Training for {args.epochs} epochs...")
    start = time.time()

    # Monitor training losses
    losses = []
    X_client_t = torch.tensor(X_client, dtype=torch.float32).to(device)

    for epoch in range(args.epochs):
        # Train one epoch
        spn.train_local(X_client, epochs=1, lr=0.01)

        # Compute current loss
        with torch.no_grad():
            loss = -spn.log_prob(X_client_t).mean().item()
            losses.append(loss)

        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f"    Epoch {epoch+1:3d}/{args.epochs}: Loss={loss:.4f}")

    elapsed = time.time() - start
    local_training_losses.append(losses)

    print(f"\n  Training complete: {elapsed:.1f}s")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Loss reduction: {losses[0] - losses[-1]:.4f}")

    # Check convergence
    if len(losses) >= 10:
        last_10_losses = losses[-10:]
        loss_std = np.std(last_10_losses)
        loss_trend = last_10_losses[-1] - last_10_losses[0]

        print(f"\n  Convergence indicators (last 10 epochs):")
        print(f"    Loss std dev: {loss_std:.4f} (lower = more stable)")
        print(f"    Loss trend:   {loss_trend:.4f} (negative = still improving)")

        converged = loss_std < 0.05 and abs(loss_trend) < 0.1
        print(f"    Converged: {'✅ YES' if converged else '⚠️  NO (still improving)'}")

    # Evaluate local SPN
    print(f"\n  {'─' * 78}")
    print(f"  EVALUATION: Local SPN Client {k}")
    print(f"  {'─' * 78}")

    result_k = evaluator.evaluate_local_spn(
        spn, client_id=k, n_samples_mmd=500, n_permutations=100
    )

    # Print key metrics
    print(f"\n  Quality Metrics:")
    print(f"    Overfitting gap: {result_k['overfitting_gap']:.3f} (threshold: <0.20)")
    print(f"    MMD p-value:     {result_k['mmd_pvalue']:.3f} (threshold: >0.05)")
    print(f"    KS failed dims:  {result_k['ks_failed_dims']}/{d} (threshold: <30%)")

    # Quality assessment
    gap_ok = result_k["overfitting_gap"] < 0.20
    mmd_ok = result_k["mmd_pvalue"] > 0.05
    ks_ok = result_k["ks_failed_dims"] < 0.3 * d

    quality = "✅ GOOD" if (gap_ok and mmd_ok and ks_ok) else "⚠️  NEEDS MORE TRAINING"
    print(f"\n  Overall quality: {quality}")

    local_spns.append(spn)

print("\n" + "=" * 80)
print("PHASE 2: GLOBAL SPN AGGREGATION & EM REFINEMENT")
print("=" * 80)

from causallearn.utils.FedPC import GlobalFedSPN

# For horizontal scenario: mixture of local SPNs
print("\nAggregating local SPNs into global model...")

# Compute mixture weights (sample-proportional)
client_counts = [len(X_splits[k]) for k in range(K)]
weights = np.array(client_counts) / sum(client_counts)

print(f"  Mixture weights (sample-proportional):")
for k in range(K):
    print(f"    Client {k}: {weights[k]:.4f} ({client_counts[k]} samples)")

# Create global SPN
global_spn = GlobalFedSPN(
    local_spns, weights=weights, strategy="mixture", device=device
)

print(f"\n  Global SPN created (mixture of {K} local SPNs)")

# EM refinement
print(f"\n  Running EM to refine mixture weights...")
X_aug_t = torch.tensor(X_aug, dtype=torch.float32).to(device)

# Compute LL before EM
with torch.no_grad():
    ll_before = global_spn.log_prob(X_aug_t).mean().item()

print(f"    LL before EM: {ll_before:.4f}")

# EM refinement
global_spn.train_weights_em(X_aug_t)

# Compute LL after EM
with torch.no_grad():
    ll_after = global_spn.log_prob(X_aug_t).mean().item()

print(f"    LL after EM:  {ll_after:.4f}")
print(f"    LL gain:      {ll_after - ll_before:.4f}")

# Show refined weights
refined_weights = global_spn.weights.cpu().numpy()
print(f"\n  Refined mixture weights:")
for k in range(K):
    print(
        f"    Client {k}: {refined_weights[k]:.4f} (delta: {refined_weights[k] - weights[k]:+.4f})"
    )

# Evaluate global SPN
print(f"\n  {'─' * 78}")
print(f"  EVALUATION: Global Federated SPN")
print(f"  {'─' * 78}")

# Wrap global SPN with routing options
from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH_SPN_Wrapper

print("\n  Evaluating with routing=False (mixture mode)...")
global_spn_no_routing = FedCDH_SPN_Wrapper(global_spn, u_index=d, routing=False)

result_no_routing = evaluator.evaluate_global_spn(
    global_spn_no_routing,
    n_samples_mmd=1000,
    n_permutations=100,
    weights=refined_weights,
    ll_before_em=ll_before,
    ll_after_em=ll_after,
)

print("\n  Evaluating with routing=True (context-aware)...")
global_spn_routing = FedCDH_SPN_Wrapper(global_spn, u_index=d, routing=True)

result_routing = evaluator.evaluate_global_spn(
    global_spn_routing,
    n_samples_mmd=1000,
    n_permutations=100,
    weights=refined_weights,
    ll_before_em=ll_before,
    ll_after_em=ll_after,
)

print(f"\n  Quality Metrics:")
print(f"    {'Metric':<25} {'routing=False':<15} {'routing=True':<15}")
print(f"    {'-'*25} {'-'*15} {'-'*15}")
print(
    f"    {'Test LL':<25} {result_no_routing['test_ll']:>14.4f} {result_routing['test_ll']:>14.4f}"
)
print(
    f"    {'MMD p-value':<25} {result_no_routing['mmd_pvalue']:>14.3f} {result_routing['mmd_pvalue']:>14.3f}"
)

# Compare to local SPNs
if len(evaluator.local_results) > 0:
    local_avg_ll = np.mean([r["test_ll"] for r in evaluator.local_results])
    print(f"    {'Local avg LL':<25} {local_avg_ll:>14.4f} {local_avg_ll:>14.4f}")
    print(
        f"    {'LL gain over locals':<25} {result_no_routing['test_ll'] - local_avg_ll:>14.4f} {result_routing['test_ll'] - local_avg_ll:>14.4f}"
    )

# Use routing=True result for overall quality
global_result = result_routing
mmd_ok = global_result["mmd_pvalue"] > 0.05
ll_ok = abs(global_result["test_ll"] - local_avg_ll) < 0.1
quality = "✅ GOOD" if (mmd_ok and ll_ok) else "⚠️  NEEDS MORE TRAINING"
print(f"\n  Overall quality (with routing=True): {quality}")
print(f"  Note: routing=True is the correct mode when context U is observed")

print("\n" + "=" * 80)
print("GENERATE REPORT")
print("=" * 80)

output_dir = Path("tests/experiments/spn_training_with_eval")
evaluator.generate_report(output_dir, include_umap=True)

print("\n" + "=" * 80)
print("CONVERGENCE SUMMARY")
print("=" * 80)

print("\nHow to tell if SPNs are well-trained:")
print("\n1. TRAINING LOSS CONVERGENCE:")
print("   ✅ Loss decreases smoothly without oscillations")
print("   ✅ Loss std dev < 0.05 in last 10 epochs (stable)")
print("   ✅ Loss trend ≈ 0 in last 10 epochs (converged)")
print("   ❌ Loss still decreasing → needs more epochs")

print("\n2. OVERFITTING GAP (Train vs Test LL):")
print("   ✅ Gap < 0.20: Good generalization")
print("   ⚠️  Gap 0.20-0.50: Mild overfitting, acceptable")
print("   ❌ Gap > 0.50: Severe overfitting, reduce model complexity")

print("\n3. DISTRIBUTION MATCHING (MMD Test):")
print("   ✅ p-value > 0.05: Generated samples match test data")
print("   ❌ p-value < 0.05: SPN not capturing data distribution")

print("\n4. MARGINAL DISTRIBUTIONS (KS Test):")
print("   ✅ Failed dims < 30%: Most dimensions match")
print("   ❌ Failed dims > 50%: Poor marginal matching")

print("\n5. GLOBAL vs LOCAL:")
print("   ✅ Global LL > weighted avg of local LLs")
print("   ✅ Global MMD p-value > local MMD p-values")
print("   → Indicates successful federated aggregation")

print("\n" + "=" * 80)
print("CURRENT RESULTS SUMMARY")
print("=" * 80)

print("\nLocal SPNs:")
for k, result in enumerate(evaluator.local_results):
    converged = (
        "✅"
        if local_training_losses[k][-1] - np.mean(local_training_losses[k][-10:-5])
        < 0.1
        else "⚠️"
    )
    gap_ok = "✅" if result["overfitting_gap"] < 0.20 else "⚠️"
    mmd_ok = "✅" if result["mmd_pvalue"] > 0.05 else "⚠️"

    print(f"  Client {k}:")
    print(f"    Convergence:     {converged}")
    print(f"    Overfitting gap: {gap_ok} {result['overfitting_gap']:.3f}")
    print(f"    MMD p-value:     {mmd_ok} {result['mmd_pvalue']:.3f}")

print("\nGlobal SPN:")
mmd_ok = "✅" if global_result["mmd_pvalue"] > 0.05 else "⚠️"
print(f"  MMD p-value: {mmd_ok} {global_result['mmd_pvalue']:.3f}")

# Final verdict
all_local_converged = all(
    result["overfitting_gap"] < 0.20 and result["mmd_pvalue"] > 0.05
    for result in evaluator.local_results
)
global_good = global_result["mmd_pvalue"] > 0.05

if all_local_converged and global_good:
    verdict = "✅ ALL SPNs WELL-TRAINED AND CONVERGED"
elif global_good:
    verdict = "⚠️  Global SPN good, local SPNs need more training"
else:
    verdict = "❌ MORE TRAINING NEEDED (increase epochs)"

print(f"\nFinal Verdict: {verdict}")

print("\n" + "=" * 80)
print(f"Report saved to: {output_dir}")
print("=" * 80)
