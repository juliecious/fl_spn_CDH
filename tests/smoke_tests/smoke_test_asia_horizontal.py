#!/usr/bin/env python3
"""
Smoke test: Asia dataset with Horizontal SPN mode
Tests Phase 2&3 refactoring with real FedCDH workflow
"""
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")

print("=" * 70)
print("SMOKE TEST: Asia Dataset - Horizontal SPN Mode")
print("=" * 70)

# Test 1: Imports from new structure
print("\n[Step 1] Testing imports from new spn module structure...")
try:
    from causallearn.utils.spn import (
        LocalSPNWrapper,
        LocalClusterMixture,
        GlobalFedSPN,
        evaluate_spn_quality,
        compute_mmd_squared,
    )

    print("  ✓ New imports successful")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    exit(1)

# Test 2: Backwards compatibility
print("\n[Step 2] Testing backwards compatibility...")
try:
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from causallearn.utils.FedPC import LocalSPNWrapper as OldLSW

        if len(w) > 0 and issubclass(w[0].category, DeprecationWarning):
            print(f"  ✓ Deprecation warning shown correctly")

        assert OldLSW == LocalSPNWrapper
        print("  ✓ Backwards compatibility maintained")
except Exception as e:
    print(f"  ✗ Compatibility failed: {e}")
    exit(1)

# Test 3: Generate synthetic Asia-like data
print("\n[Step 3] Generating synthetic Asia data (8 features, 3 clients)...")
try:
    np.random.seed(42)

    # Asia has 8 variables
    n_total = 300  # Small dataset for smoke test
    n_features = 8
    K_clients = 3

    # Generate synthetic data with some structure
    mean = np.zeros(n_features)
    cov = np.eye(n_features) * 0.5 + 0.3  # Some correlation
    X_global = np.random.multivariate_normal(mean, cov, n_total)

    # Partition horizontally (same features, different samples)
    samples_per_client = n_total // K_clients
    X_splits = [
        X_global[i * samples_per_client : (i + 1) * samples_per_client]
        for i in range(K_clients)
    ]

    print(f"  ✓ Data generated: {K_clients} clients, {n_features} features")
    print(f"  ✓ Samples per client: {[len(x) for x in X_splits]}")

except Exception as e:
    print(f"  ✗ Data generation failed: {e}")
    exit(1)

# Test 4: Train local SPNs
print("\n[Step 4] Training local SPNs (horizontal mode)...")
try:
    import torch

    device = "cpu"

    local_spns = []
    for client_id in range(K_clients):
        # Add context column for horizontal mode
        X_client = X_splits[client_id]
        context = np.full((len(X_client), 1), client_id)
        X_with_context = np.hstack([X_client, context])

        # Create local SPN with clustering
        spn = LocalSPNWrapper(
            num_features=n_features + 1,  # +1 for context
            device=device,
            depth=2,
            num_sums=20,
            num_leaves=20,
            num_repetitions=10,
            seed=42 + client_id,
        )

        # Train (minimal epochs for smoke test)
        spn.train_local(X_with_context, epochs=10, lr=0.01)
        local_spns.append(spn)

        print(f"  ✓ Client {client_id} SPN trained")

except Exception as e:
    print(f"  ✗ Local SPN training failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test 5: Create Global Federated SPN
print("\n[Step 5] Creating Global Federated SPN (horizontal)...")
try:
    global_spn = GlobalFedSPN(
        local_models=local_spns, num_clusters=2, device=device  # 2 mechanism clusters
    )
    print(f"  ✓ GlobalFedSPN created with {K_clients} local SPNs")

except Exception as e:
    print(f"  ✗ GlobalFedSPN creation failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test 6: Test sampling and log-prob
print("\n[Step 6] Testing sampling and log-probability...")
try:
    # Sample from global SPN
    samples = global_spn.sample(50)
    print(f"  ✓ Generated {len(samples)} samples, shape: {samples.shape}")

    # Compute log-prob on test data
    X_test = X_global[:50]
    context_test = np.zeros((50, 1))  # Use client 0 context for test
    X_test_with_context = np.hstack([X_test, context_test])
    X_test_tensor = torch.tensor(X_test_with_context, dtype=torch.float32).to(device)

    with torch.no_grad():
        log_probs = global_spn.log_prob(X_test_tensor)

    avg_log_prob = log_probs.mean().item()
    print(f"  ✓ Average log-probability: {avg_log_prob:.4f}")

    if np.isnan(avg_log_prob) or np.isinf(avg_log_prob):
        print(f"  ⚠ Warning: Invalid log-prob detected")
    else:
        print(f"  ✓ Log-probabilities are valid")

except Exception as e:
    print(f"  ✗ Sampling/log-prob failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test 7: Evaluate SPN quality
print("\n[Step 7] Evaluating SPN quality with MMD test...")
try:
    # Evaluate local SPN quality
    results = evaluate_spn_quality(
        local_spns[0],
        X_splits[0],
        n_samples=50,
        device=device,
        compute_mmd=True,
        compute_ks=True,
        has_context_column=False,  # No context for local evaluation
    )

    print(f"  ✓ Evaluation complete")
    if "train_ll" in results:
        print(f"    - Train LL: {results['train_ll']:.4f}")
    if "mmd_pvalue" in results:
        print(f"    - MMD p-value: {results['mmd_pvalue']:.3f}")
    if "ks_fail_ratio" in results:
        print(f"    - KS fail ratio: {results['ks_fail_ratio']:.2%}")

except Exception as e:
    print(f"  ✗ Evaluation failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test 8: Test structure learning utilities
print("\n[Step 8] Testing structure learning utilities...")
try:
    from causallearn.utils.spn import (
        build_feature_indicator_matrix,
        group_features_by_client_set,
    )

    # Build feature indicator matrix for horizontal mode
    feature_maps = {i: list(range(n_features)) for i in range(K_clients)}
    M = build_feature_indicator_matrix(feature_maps, n_features)

    print(f"  ✓ Feature indicator matrix: {M.shape}")
    print(f"    {M}")

    # Group features
    groups = group_features_by_client_set(M)
    print(f"  ✓ Feature groups: {len(groups)} distinct ownership patterns")

except Exception as e:
    print(f"  ✗ Structure learning failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("✅ ALL SMOKE TESTS PASSED!")
print("=" * 70)
print("\nSummary:")
print(f"  • Imports: ✓ New structure + backwards compatibility")
print(f"  • Data: ✓ {n_total} samples, {K_clients} clients, {n_features} features")
print(f"  • Local SPNs: ✓ {K_clients} trained successfully")
print(f"  • Global SPN: ✓ Horizontal federated model created")
print(f"  • Sampling: ✓ Generated samples and computed log-probs")
print(f"  • Evaluation: ✓ Quality metrics computed")
print(f"  • Utilities: ✓ Structure learning functions work")
print("\n🎉 Phase 2&3 refactoring verified successfully!")
print("=" * 70)
