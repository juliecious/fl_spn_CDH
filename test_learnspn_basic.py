"""
Test SPFlow LearnSPN basic functionality
"""
import numpy as np
import torch

print("=" * 80)
print("TESTING SPFLOW LEARNSPN")
print("=" * 80)
print()

# Test 1: Import SPFlow
print("Test 1: Importing SPFlow...")
try:
    from spflow.learn import learn_spn
    from spflow.modules.leaves import Normal

    print("✅ SPFlow imported successfully")
    print(f"  learn_spn function: {learn_spn}")
    print(f"  Normal leaf: {Normal}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print()

# Test 2: Generate simple data
print("Test 2: Generating test data...")
np.random.seed(42)

# Simple 2D Gaussian mixture
n = 500
d = 2

# Cluster 1
X1 = np.random.randn(n // 2, d) + np.array([2, 2])
# Cluster 2
X2 = np.random.randn(n // 2, d) + np.array([-2, -2])
X_train = np.vstack([X1, X2])

print(f"  Data shape: {X_train.shape}")
print(f"  Mean: {X_train.mean(axis=0)}")
print(f"  Std: {X_train.std(axis=0)}")
print()

# Test 3: Learn SPN structure
print("Test 3: Learning SPN structure with LearnSPN...")
try:
    # Convert to torch tensor
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)

    # Create Normal leaf for each feature
    leaf = Normal(out_channels=1)

    # SPFlow learn_spn - structure learning
    print("  Attempting structure learning...")
    spn = learn_spn(
        X_train_torch,
        leaf_modules=leaf,
        min_instances_slice=50,  # Min samples to split (less strict)
        min_features_slice=1,  # Allow single features
    )
    print("✅ SPN structure learned")
    print(f"  Type: {type(spn)}")
except Exception as e:
    print(f"❌ Structure learning failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print()

# Test 4: Evaluate log-likelihood
print("Test 4: Computing log-likelihood...")
try:
    from spflow.algorithms.inference.Inference import log_likelihood

    ll_train = log_likelihood(spn, X_train)
    mean_ll = np.mean(ll_train)

    print(f"  Train LL (per sample): {mean_ll:.4f}")
    print(f"  LL range: [{ll_train.min():.2f}, {ll_train.max():.2f}]")

    # Compare with simple Gaussian
    cov = np.cov(X_train.T)
    simple_ll = -0.5 * (
        d * np.log(2 * np.pi)
        + np.log(np.linalg.det(cov))
        + np.mean(
            [
                (x - X_train.mean(axis=0))
                @ np.linalg.inv(cov)
                @ (x - X_train.mean(axis=0))
                for x in X_train
            ]
        )
    )

    print(f"  Simple Gaussian LL: {simple_ll:.4f}")
    print(f"  LearnSPN improvement: {mean_ll - simple_ll:.4f}")

    if mean_ll > simple_ll:
        print("  ✅ LearnSPN better than simple Gaussian")
    else:
        print("  ⚠️  Simple Gaussian is better (unexpected)")

except Exception as e:
    print(f"❌ Evaluation failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print()

# Test 5: Sample from learned SPN
print("Test 5: Sampling from learned SPN...")
try:
    from spflow.algorithms.sampling.Sampling import sample_instances

    samples = sample_instances(spn, np.zeros((100, d)), seed=42)

    print(f"  Samples shape: {samples.shape}")
    print(f"  Samples mean: {samples.mean(axis=0)}")
    print(f"  Samples std: {samples.std(axis=0)}")
    print("  ✅ Sampling works")

except Exception as e:
    print(f"❌ Sampling failed: {e}")
    import traceback

    traceback.print_exc()

print()

# Test 6: Check structure
print("Test 6: Analyzing learned structure...")
try:

    def count_nodes(node, node_type=None):
        """Count nodes in SPN tree."""
        from spflow.structure.Base import Sum, Product, Leaf

        if isinstance(node, Leaf):
            return 1 if node_type == "leaf" else 0

        count = 0
        if node_type == "sum" and isinstance(node, Sum):
            count = 1
        elif node_type == "product" and isinstance(node, Product):
            count = 1

        if hasattr(node, "children"):
            for child in node.children:
                count += count_nodes(child, node_type)

        return count

    n_sums = count_nodes(spn, "sum")
    n_products = count_nodes(spn, "product")
    n_leaves = count_nodes(spn, "leaf")

    print(f"  Sum nodes: {n_sums}")
    print(f"  Product nodes: {n_products}")
    print(f"  Leaf nodes: {n_leaves}")
    print(f"  Total depth: ~{n_sums + n_products}")

except Exception as e:
    print(f"⚠️  Structure analysis failed: {e}")

print()
print("=" * 80)
print("BASIC TEST COMPLETED")
print("=" * 80)
print()
print("Next steps:")
print("1. Create LearnSPNWrapper for FedCDH")
print("2. Run comparison experiment with RAT-SPN")
print("3. Analyze performance differences")
