#!/usr/bin/env python
"""
Quick test: Verify ANM-based orientation works for cross-client edges.
Tests the compute_anm_score function directly without running full FedCDH.
"""

import numpy as np
from causallearn.utils.mechanism_invariance import compute_anm_score

print("=" * 70)
print("ANM ORIENTATION QUICK TEST")
print("=" * 70)

# Test 1: Clear causal relationship X → Y
print("\nTest 1: X → Y (Y = 2*X + noise)")
np.random.seed(42)
n = 500
X = np.random.randn(n, 3)
X[:, 1] = 2 * X[:, 0] + 0.3 * np.random.randn(n)  # Feature 0 → Feature 1

score_0_to_1 = compute_anm_score(1, [0], X, num_samples=500)
score_1_to_0 = compute_anm_score(0, [1], X, num_samples=500)

print(f"  Score 0→1: {score_0_to_1:.4f}")
print(f"  Score 1→0: {score_1_to_0:.4f}")
print(f"  Correct direction: 0→1")
print(f"  Selected direction: {'0→1' if score_0_to_1 > score_1_to_0 else '1→0'}")
print(f"  ✓ PASS" if score_0_to_1 > score_1_to_0 else "  ✗ FAIL")

# Test 2: Nonlinear relationship X → Y
print("\nTest 2: X → Y (Y = X² + noise)")
np.random.seed(43)
X2 = np.random.randn(n, 3)
X2[:, 1] = X2[:, 0] ** 2 + 0.5 * np.random.randn(n)  # Feature 0 → Feature 1

score_0_to_1_nl = compute_anm_score(1, [0], X2, num_samples=500)
score_1_to_0_nl = compute_anm_score(0, [1], X2, num_samples=500)

print(f"  Score 0→1: {score_0_to_1_nl:.4f}")
print(f"  Score 1→0: {score_1_to_0_nl:.4f}")
print(f"  Correct direction: 0→1")
print(f"  Selected direction: {'0→1' if score_0_to_1_nl > score_1_to_0_nl else '1→0'}")
print(f"  ✓ PASS" if score_0_to_1_nl > score_1_to_0_nl else "  ✗ FAIL")

# Test 3: Reverse relationship Y → X
print("\nTest 3: Y → X (X = 1.5*Y + noise)")
np.random.seed(44)
X3 = np.random.randn(n, 3)
X3[:, 0] = 1.5 * X3[:, 1] + 0.4 * np.random.randn(n)  # Feature 1 → Feature 0

score_0_to_1_rev = compute_anm_score(1, [0], X3, num_samples=500)
score_1_to_0_rev = compute_anm_score(0, [1], X3, num_samples=500)

print(f"  Score 0→1: {score_0_to_1_rev:.4f}")
print(f"  Score 1→0: {score_1_to_0_rev:.4f}")
print(f"  Correct direction: 1→0")
print(
    f"  Selected direction: {'0→1' if score_0_to_1_rev > score_1_to_0_rev else '1→0'}"
)
print(f"  ✓ PASS" if score_1_to_0_rev > score_0_to_1_rev else "  ✗ FAIL")

print("\n" + "=" * 70)
print("ANM ORIENTATION TEST COMPLETE")
print("=" * 70)
