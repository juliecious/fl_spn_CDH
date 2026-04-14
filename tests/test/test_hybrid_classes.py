"""
Test suite for hybrid mode Mixture-then-Product classes.

Tests GroupMixture, ProductOverGroups, and ProductOverGroupsWithOverlap
as defined in Seng et al. (2025).
"""
import sys
import os
import numpy as np
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from causallearn.utils.FedPC import (
    GroupMixture,
    ProductOverGroups,
    ProductOverGroupsWithOverlap,
    LocalSPNWrapper,
    UnivariateSPNWrapper,
)


# ============================================================
# Test 1: GroupMixture Initialization
# ============================================================


def test_groupmixture_init_basic():
    """Test 1.1: Basic initialization with 2 clients."""
    # Create two SPNs for a 3-feature group (depth=1 since 2^1=2 <= 3)
    spn0 = LocalSPNWrapper(
        num_features=3, device="cpu", num_sums=5, num_leaves=5, depth=1
    )
    spn1 = LocalSPNWrapper(
        num_features=3, device="cpu", num_sums=5, num_leaves=5, depth=1
    )

    # Train on dummy data to initialize parameters
    dummy_data = np.random.randn(50, 3)
    spn0.train_local(dummy_data, epochs=5)
    spn1.train_local(dummy_data, epochs=5)

    # Create mixture
    mixture = GroupMixture(
        [spn0, spn1], weights=[0.4, 0.6], feature_indices=[0, 1, 2], device="cpu"
    )

    # Assertions
    assert len(mixture.client_spns) == 2, "Should have 2 client SPNs"
    assert torch.allclose(
        mixture.weights.sum(), torch.tensor(1.0)
    ), "Weights must sum to 1"
    assert mixture.feature_indices == [0, 1, 2], "Feature indices mismatch"
    print("✓ Test 1.1 passed: Basic initialization")


def test_groupmixture_init_validation():
    """Test 1.2: Initialization validation checks."""
    spn0 = LocalSPNWrapper(num_features=2, device="cpu", depth=1)

    # Test 1: Weights don't sum to 1
    try:
        GroupMixture([spn0], weights=[0.5], feature_indices=[0, 1])
        assert False, "Should raise AssertionError for invalid weights"
    except AssertionError as e:
        assert "Weights must sum to 1" in str(e)

    # Test 2: Mismatched SPNs and weights
    spn1 = LocalSPNWrapper(num_features=2, device="cpu", depth=1)
    try:
        GroupMixture([spn0, spn1], weights=[1.0], feature_indices=[0, 1])
        assert False, "Should raise AssertionError for mismatched counts"
    except AssertionError as e:
        assert "Mismatched SPNs" in str(e)

    # Test 3: Empty feature indices
    try:
        GroupMixture([spn0], weights=[1.0], feature_indices=[])
        assert False, "Should raise AssertionError for empty indices"
    except AssertionError as e:
        assert "Feature indices cannot be empty" in str(e)

    print("✓ Test 1.2 passed: Validation checks")


# ============================================================
# Test 2: GroupMixture log_prob
# ============================================================


def test_groupmixture_log_prob_identical_spns():
    """Test 2.1: Mixture of identical SPNs equals single SPN."""
    # Create one SPN (depth=1 for 3 features)
    spn = LocalSPNWrapper(
        num_features=3, device="cpu", num_sums=5, num_leaves=5, depth=1, seed=42
    )

    # Train on data
    np.random.seed(42)
    train_data = np.random.randn(100, 3)
    spn.train_local(train_data, epochs=10, lr=0.01)

    # Create mixture with same SPN twice (equal weights)
    mixture = GroupMixture(
        [spn, spn], weights=[0.5, 0.5], feature_indices=[0, 1, 2], device="cpu"
    )

    # Test data (5D, but we only use first 3)
    test_data = torch.randn(50, 5)
    test_data_3d = test_data[:, [0, 1, 2]]

    # Compute log-probs
    mixture_ll = mixture.log_prob(test_data)  # Uses features [0,1,2]
    single_ll = spn.log_prob(test_data_3d)  # Direct evaluation

    # They should be equal (mixture of identical components)
    assert mixture_ll.shape == (50, 1), "Shape mismatch"
    assert torch.allclose(mixture_ll, single_ll, atol=1e-4), (
        f"Mixture of identical SPNs should equal single SPN\n"
        f"Mixture: {mixture_ll[:5].squeeze()}\n"
        f"Single:  {single_ll[:5].squeeze()}"
    )

    print("✓ Test 2.1 passed: Identical SPNs property")


def test_groupmixture_log_prob_feature_extraction():
    """Test 2.2: Correct feature extraction from full data matrix."""
    # Create SPN for features [2, 3] (middle of 5-feature space, depth=1 for 2 features)
    spn0 = LocalSPNWrapper(
        num_features=2, device="cpu", num_sums=5, num_leaves=5, depth=1, seed=10
    )
    spn1 = LocalSPNWrapper(
        num_features=2, device="cpu", num_sums=5, num_leaves=5, depth=1, seed=20
    )

    # Train on 2D data
    np.random.seed(42)
    train_data = np.random.randn(80, 2)
    spn0.train_local(train_data, epochs=5)
    spn1.train_local(train_data, epochs=5)

    # Mixture for features [2, 3]
    mixture = GroupMixture(
        [spn0, spn1], weights=[0.3, 0.7], feature_indices=[2, 3], device="cpu"
    )

    # Full 5D test data
    test_data_5d = torch.randn(30, 5)

    # Compute log-prob via mixture (extracts [:, [2,3]])
    mixture_ll = mixture.log_prob(test_data_5d)

    # Manually compute: weighted mixture over features [2,3]
    test_data_2d = test_data_5d[:, [2, 3]]
    ll0 = spn0.log_prob(test_data_2d)
    ll1 = spn1.log_prob(test_data_2d)
    manual_ll = torch.logsumexp(
        torch.cat([ll0 + np.log(0.3), ll1 + np.log(0.7)], dim=1), dim=1, keepdim=True
    )

    assert mixture_ll.shape == (30, 1), "Shape mismatch"
    assert torch.allclose(
        mixture_ll, manual_ll, atol=1e-4
    ), "Feature extraction or mixture computation incorrect"

    print("✓ Test 2.2 passed: Feature extraction")


# ============================================================
# Test 3: GroupMixture sample
# ============================================================


def test_groupmixture_sample_shape():
    """Test 3.1: Sample returns correct shape."""
    # Create mixture for 3 features (depth=1)
    spn0 = LocalSPNWrapper(
        num_features=3, device="cpu", num_sums=5, num_leaves=5, depth=1, seed=1
    )
    spn1 = LocalSPNWrapper(
        num_features=3, device="cpu", num_sums=5, num_leaves=5, depth=1, seed=2
    )

    # Train
    train_data = np.random.randn(60, 3)
    spn0.train_local(train_data, epochs=5)
    spn1.train_local(train_data, epochs=5)

    mixture = GroupMixture([spn0, spn1], weights=[0.5, 0.5], feature_indices=[0, 1, 2])

    # Sample
    samples = mixture.sample(100)

    assert samples.shape == (100, 3), f"Expected (100, 3), got {samples.shape}"
    assert not torch.isnan(samples).any(), "Samples contain NaN"
    print("✓ Test 3.1 passed: Sample shape")


def test_groupmixture_sample_distribution():
    """Test 3.2: Samples approximately match mixture distribution."""
    # Create two well-separated Gaussian SPNs (depth=1 for 2 features)
    spn0 = LocalSPNWrapper(
        num_features=2, device="cpu", num_sums=5, num_leaves=5, depth=1, seed=10
    )
    spn1 = LocalSPNWrapper(
        num_features=2, device="cpu", num_sums=5, num_leaves=5, depth=1, seed=20
    )

    # Train on distinct distributions
    train_data0 = np.random.randn(100, 2) + np.array([0, 0])  # Mean [0, 0]
    train_data1 = np.random.randn(100, 2) + np.array([5, 5])  # Mean [5, 5]

    spn0.train_local(train_data0, epochs=20)
    spn1.train_local(train_data1, epochs=20)

    # Mixture with 70% from spn1 (mean [5,5])
    mixture = GroupMixture([spn0, spn1], weights=[0.3, 0.7], feature_indices=[0, 1])

    # Sample
    samples = mixture.sample(1000)

    # Check: mean should be closer to [5, 5] than [0, 0] (due to 70% weight)
    sample_mean = samples.mean(dim=0).numpy()
    expected_mean = 0.3 * np.array([0, 0]) + 0.7 * np.array([5, 5])  # [3.5, 3.5]

    distance_to_expected = np.linalg.norm(sample_mean - expected_mean)
    assert (
        distance_to_expected < 1.5
    ), f"Sample mean {sample_mean} too far from expected {expected_mean}"

    print("✓ Test 3.2 passed: Sample distribution")


# ============================================================
# Test 4: GroupMixture with UnivariateSPNWrapper
# ============================================================


def test_groupmixture_univariate():
    """Test 4.1: GroupMixture with 1D features (UnivariateSPNWrapper)."""
    # Create univariate SPNs
    spn0 = UnivariateSPNWrapper(device="cpu", num_leaves=10, seed=5)
    spn1 = UnivariateSPNWrapper(device="cpu", num_leaves=10, seed=6)

    # Train on 1D data
    train_data0 = np.random.randn(80, 1)
    train_data1 = np.random.randn(80, 1) + 2.0  # Shifted distribution

    spn0.train_local(train_data0, epochs=10)
    spn1.train_local(train_data1, epochs=10)

    # Create mixture for feature [3] in 5D space
    mixture = GroupMixture([spn0, spn1], weights=[0.5, 0.5], feature_indices=[3])

    # Test log_prob
    test_data = torch.randn(40, 5)
    log_p = mixture.log_prob(test_data)

    assert log_p.shape == (40, 1), "Shape mismatch"
    assert torch.isfinite(log_p).all(), "Non-finite log probabilities"

    # Test sample
    samples = mixture.sample(50)
    assert samples.shape == (50, 1), "Sample shape mismatch"

    print("✓ Test 4.1 passed: Univariate mixture")


# ============================================================
# Run all tests
# ============================================================


# ============================================================
# Test 5: ProductOverGroups Initialization
# ============================================================


def test_product_init_basic():
    """Test 5.1: Basic initialization with 2 disjoint groups."""
    # Create SPNs for two groups
    spn0_g0 = LocalSPNWrapper(
        num_features=3, device="cpu", num_sums=5, num_leaves=5, depth=1, seed=1
    )
    spn1_g0 = LocalSPNWrapper(
        num_features=3, device="cpu", num_sums=5, num_leaves=5, depth=1, seed=2
    )
    spn0_g1 = LocalSPNWrapper(
        num_features=2, device="cpu", num_sums=5, num_leaves=5, depth=1, seed=3
    )
    spn1_g1 = LocalSPNWrapper(
        num_features=2, device="cpu", num_sums=5, num_leaves=5, depth=1, seed=4
    )

    # Train on dummy data
    train_data_g0 = np.random.randn(50, 3)
    train_data_g1 = np.random.randn(50, 2)
    spn0_g0.train_local(train_data_g0, epochs=5)
    spn1_g0.train_local(train_data_g0, epochs=5)
    spn0_g1.train_local(train_data_g1, epochs=5)
    spn1_g1.train_local(train_data_g1, epochs=5)

    # Create mixtures for each group
    mix_g0 = GroupMixture(
        [spn0_g0, spn1_g0], weights=[0.5, 0.5], feature_indices=[0, 1, 2]
    )
    mix_g1 = GroupMixture(
        [spn0_g1, spn1_g1], weights=[0.5, 0.5], feature_indices=[3, 4]
    )

    # Create product
    product = ProductOverGroups([mix_g0, mix_g1], [[0, 1, 2], [3, 4]], device="cpu")

    # Assertions
    assert len(product.group_mixtures) == 2, "Should have 2 group mixtures"
    assert product.feature_groups == [[0, 1, 2], [3, 4]], "Feature groups mismatch"
    assert product.num_features == 5, "Total features should be 5"
    print("✓ Test 5.1 passed: Basic initialization")


def test_product_overlap_detection():
    """Test 5.2: Detect overlapping feature groups."""
    # Create SPNs
    spn0 = LocalSPNWrapper(num_features=3, device="cpu", depth=1)
    spn1 = LocalSPNWrapper(num_features=3, device="cpu", depth=1)

    # Create mixtures with overlapping features
    mix_g0 = GroupMixture([spn0], weights=[1.0], feature_indices=[0, 1, 2])
    mix_g1 = GroupMixture([spn1], weights=[1.0], feature_indices=[1, 2, 3])  # Overlap!

    # Should raise ValueError
    try:
        product = ProductOverGroups([mix_g0, mix_g1], [[0, 1, 2], [1, 2, 3]])
        assert False, "Should raise ValueError for overlapping groups"
    except ValueError as e:
        assert "overlap" in str(e).lower()
        assert "[1, 2]" in str(e), "Should identify which features overlap"

    print("✓ Test 5.2 passed: Overlap detection")


# ============================================================
# Test 6: ProductOverGroups log_prob
# ============================================================


def test_product_log_prob_sum():
    """Test 6.1: log P(X) equals sum of group log probs."""
    # Create product of two groups
    spn0_g0 = LocalSPNWrapper(num_features=3, device="cpu", depth=1, seed=10)
    spn0_g1 = LocalSPNWrapper(num_features=2, device="cpu", depth=1, seed=20)

    # Train
    train_g0 = np.random.randn(80, 3)
    train_g1 = np.random.randn(80, 2)
    spn0_g0.train_local(train_g0, epochs=10)
    spn0_g1.train_local(train_g1, epochs=10)

    # Create mixtures (single SPN each for simplicity)
    mix_g0 = GroupMixture([spn0_g0], weights=[1.0], feature_indices=[0, 1, 2])
    mix_g1 = GroupMixture([spn0_g1], weights=[1.0], feature_indices=[3, 4])

    # Create product
    product = ProductOverGroups([mix_g0, mix_g1], [[0, 1, 2], [3, 4]])

    # Test data
    test_data = torch.randn(50, 5)

    # Compute log-prob via product
    product_ll = product.log_prob(test_data)

    # Manual computation: log P(X) = log P(X_g0) + log P(X_g1)
    manual_ll = mix_g0.log_prob(test_data) + mix_g1.log_prob(test_data)

    assert product_ll.shape == (50, 1), "Shape mismatch"
    assert torch.allclose(product_ll, manual_ll, atol=1e-5), (
        f"Product log-prob should equal sum of group log-probs\n"
        f"Product: {product_ll[:3].squeeze()}\n"
        f"Manual:  {manual_ll[:3].squeeze()}"
    )

    print("✓ Test 6.1 passed: log_prob is sum of groups")


def test_product_log_prob_independence():
    """Test 6.2: Groups evaluated independently (no information leakage)."""
    # Create two groups with different SPNs
    spn0_g0 = LocalSPNWrapper(num_features=2, device="cpu", depth=1, seed=5)
    spn0_g1 = LocalSPNWrapper(num_features=2, device="cpu", depth=1, seed=6)

    # Train on different distributions
    train_g0 = np.random.randn(60, 2) + np.array([0, 0])
    train_g1 = np.random.randn(60, 2) + np.array([5, 5])
    spn0_g0.train_local(train_g0, epochs=10)
    spn0_g1.train_local(train_g1, epochs=10)

    mix_g0 = GroupMixture([spn0_g0], weights=[1.0], feature_indices=[0, 1])
    mix_g1 = GroupMixture([spn0_g1], weights=[1.0], feature_indices=[2, 3])

    product = ProductOverGroups([mix_g0, mix_g1], [[0, 1], [2, 3]])

    # Create data where group 0 is from training dist, group 1 is random
    test_data = torch.randn(30, 4)
    test_data[:, [0, 1]] = torch.tensor(
        train_g0[:30], dtype=torch.float32
    )  # Good match
    test_data[:, [2, 3]] = torch.randn(30, 2) - 10  # Poor match

    # Product log-prob should be sum (group 0 high, group 1 low)
    product_ll = product.log_prob(test_data)
    g0_ll = mix_g0.log_prob(test_data)
    g1_ll = mix_g1.log_prob(test_data)

    # Verify independence: product = g0 + g1
    assert torch.allclose(
        product_ll, g0_ll + g1_ll, atol=1e-5
    ), "Groups not independent"

    print("✓ Test 6.2 passed: Group independence")


# ============================================================
# Test 7: ProductOverGroups sample
# ============================================================


def test_product_sample_shape():
    """Test 7.1: Sample returns correct shape."""
    # Create product
    spn0 = LocalSPNWrapper(num_features=3, device="cpu", depth=1, seed=1)
    spn1 = LocalSPNWrapper(num_features=2, device="cpu", depth=1, seed=2)

    train_g0 = np.random.randn(50, 3)
    train_g1 = np.random.randn(50, 2)
    spn0.train_local(train_g0, epochs=5)
    spn1.train_local(train_g1, epochs=5)

    mix_g0 = GroupMixture([spn0], weights=[1.0], feature_indices=[0, 1, 2])
    mix_g1 = GroupMixture([spn1], weights=[1.0], feature_indices=[3, 4])

    product = ProductOverGroups([mix_g0, mix_g1], [[0, 1, 2], [3, 4]])

    # Sample
    samples = product.sample(100)

    assert samples.shape == (100, 5), f"Expected (100, 5), got {samples.shape}"
    assert not torch.isnan(samples).any(), "Samples contain NaN"
    print("✓ Test 7.1 passed: Sample shape")


def test_product_sample_independence():
    """Test 7.2: Group samples are independent."""
    # Create product with deterministic seeds
    spn0 = LocalSPNWrapper(num_features=2, device="cpu", depth=1, seed=10)
    spn1 = LocalSPNWrapper(num_features=2, device="cpu", depth=1, seed=20)

    train_g0 = np.random.randn(60, 2)
    train_g1 = np.random.randn(60, 2)
    spn0.train_local(train_g0, epochs=10)
    spn1.train_local(train_g1, epochs=10)

    mix_g0 = GroupMixture([spn0], weights=[1.0], feature_indices=[0, 1])
    mix_g1 = GroupMixture([spn1], weights=[1.0], feature_indices=[2, 3])

    product = ProductOverGroups([mix_g0, mix_g1], [[0, 1], [2, 3]])

    # Sample multiple times
    samples_a = product.sample(50)
    samples_b = product.sample(50)

    # Samples should be different (stochastic)
    assert not torch.allclose(
        samples_a, samples_b
    ), "Samples are deterministic (should be random)"

    # Each group's marginal should match the group mixture's distribution
    # (statistical test: mean should be close)
    samples_large = product.sample(1000)
    mean_g0 = samples_large[:, [0, 1]].mean(dim=0)
    mean_g1 = samples_large[:, [2, 3]].mean(dim=0)

    # Means should be close to training data means (rough check)
    train_mean_g0 = train_g0.mean(axis=0)
    train_mean_g1 = train_g1.mean(axis=0)

    dist_g0 = np.linalg.norm(mean_g0.numpy() - train_mean_g0)
    dist_g1 = np.linalg.norm(mean_g1.numpy() - train_mean_g1)

    assert dist_g0 < 0.5, f"Group 0 mean mismatch: {dist_g0}"
    assert dist_g1 < 0.5, f"Group 1 mean mismatch: {dist_g1}"

    print("✓ Test 7.2 passed: Sample independence")


# ============================================================
# Test 8: ProductOverGroupsWithOverlap Overlap Detection
# ============================================================


def test_overlap_detection_basic():
    """Test 8.1: Detect overlapping features in groups."""
    # Create SPNs
    spn0 = LocalSPNWrapper(num_features=3, device="cpu", depth=1, seed=1)
    spn1 = LocalSPNWrapper(num_features=3, device="cpu", depth=1, seed=2)

    train_data = np.random.randn(50, 3)
    spn0.train_local(train_data, epochs=5)
    spn1.train_local(train_data, epochs=5)

    # Create mixtures with overlapping features [1, 2]
    mix_g0 = GroupMixture([spn0], weights=[1.0], feature_indices=[0, 1, 2])
    mix_g1 = GroupMixture([spn1], weights=[1.0], feature_indices=[1, 2, 3])

    # Should detect overlaps
    product = ProductOverGroupsWithOverlap(
        [mix_g0, mix_g1], [[0, 1, 2], [1, 2, 3]], allow_overlap=True
    )

    # Verify overlap detection
    assert product.overlap_info["has_overlap"], "Should detect overlaps"
    assert 1 in product.overlap_info["overlapping_features"], "Feature 1 overlaps"
    assert 2 in product.overlap_info["overlapping_features"], "Feature 2 overlaps"
    assert 0 not in product.overlap_info["overlapping_features"], "Feature 0 unique"
    assert 3 not in product.overlap_info["overlapping_features"], "Feature 3 unique"

    print("✓ Test 8.1 passed: Overlap detection")


def test_overlap_detection_allowed():
    """Test 8.2: allow_overlap parameter controls validation."""
    spn0 = LocalSPNWrapper(num_features=2, device="cpu", depth=1)
    spn1 = LocalSPNWrapper(num_features=2, device="cpu", depth=1)

    mix_g0 = GroupMixture([spn0], weights=[1.0], feature_indices=[0, 1])
    mix_g1 = GroupMixture([spn1], weights=[1.0], feature_indices=[1, 2])

    # With allow_overlap=True: should succeed with warning
    product_allowed = ProductOverGroupsWithOverlap(
        [mix_g0, mix_g1], [[0, 1], [1, 2]], allow_overlap=True
    )
    assert product_allowed.overlap_info["has_overlap"], "Should detect overlap"

    # With allow_overlap=False: should raise ValueError
    try:
        product_disallowed = ProductOverGroupsWithOverlap(
            [mix_g0, mix_g1], [[0, 1], [1, 2]], allow_overlap=False
        )
        assert False, "Should raise ValueError when allow_overlap=False"
    except ValueError as e:
        assert "overlap" in str(e).lower()

    print("✓ Test 8.2 passed: allow_overlap validation")


# ============================================================
# Test 9: ProductOverGroupsWithOverlap Algorithm 1 Pattern
# ============================================================


def test_overlap_algorithm1_pattern():
    """
    Test 9.1: Correct construction pattern (Algorithm 1 from Seng et al.).

    Scenario (mimicking paper's Algorithm 1):
        Client 0 has features [0, 1, 2]
        Client 1 has features [1, 2, 3]

        Correct construction (Algorithm 1):
            Group A: [0] (only client 0) → SPN_0
            Group B: [1, 2] (both clients) → Mixture(SPN_0, SPN_1)
            Group C: [3] (only client 1) → SPN_1

        Result: P(X) = P(X_0) × P(X_{1,2}) × P(X_3)
                     = P_0(X_0) × [w_0×P_0(X_{1,2}) + w_1×P_1(X_{1,2})] × P_1(X_3)

        Key: Features [1,2] appear in ONLY ONE group (B), preventing double-counting.
    """
    np.random.seed(50)
    torch.manual_seed(50)

    # Generate data
    X = np.random.randn(100, 4)

    # Client 0: Train SPNs for [0] and [1,2]
    spn_c0_A = UnivariateSPNWrapper(
        device="cpu", num_leaves=10, seed=10
    )  # 1D → univariate
    spn_c0_B = LocalSPNWrapper(num_features=2, device="cpu", depth=1, seed=11)
    spn_c0_A.train_local(X[:, [0]], epochs=10)
    spn_c0_B.train_local(X[:, [1, 2]], epochs=10)

    # Client 1: Train SPNs for [1,2] and [3]
    spn_c1_B = LocalSPNWrapper(num_features=2, device="cpu", depth=1, seed=20)
    spn_c1_C = UnivariateSPNWrapper(device="cpu", num_leaves=10, seed=21)
    spn_c1_B.train_local(X[:, [1, 2]], epochs=10)
    spn_c1_C.train_local(X[:, [3]], epochs=10)

    # Construct groups following Algorithm 1
    # Group A: [0] (only client 0)
    mix_A = GroupMixture([spn_c0_A], weights=[1.0], feature_indices=[0])

    # Group B: [1, 2] (both clients - MIXTURE)
    mix_B = GroupMixture(
        [spn_c0_B, spn_c1_B], weights=[0.5, 0.5], feature_indices=[1, 2]
    )

    # Group C: [3] (only client 1)
    mix_C = GroupMixture([spn_c1_C], weights=[1.0], feature_indices=[3])

    # Create product (should have NO overlaps after correct construction)
    product = ProductOverGroupsWithOverlap(
        [mix_A, mix_B, mix_C], [[0], [1, 2], [3]], allow_overlap=True
    )

    # Verify no overlaps (construction followed Algorithm 1 correctly)
    assert not product.overlap_info["has_overlap"], (
        "Algorithm 1 construction should have NO overlaps. "
        "Each feature appears in exactly one group."
    )

    # Test inference
    test_data = torch.tensor(X[:20], dtype=torch.float32)
    log_p = product.log_prob(test_data)

    assert log_p.shape == (20, 1), "Shape mismatch"
    assert torch.isfinite(log_p).all(), "Non-finite log probabilities"

    # Verify product property: log P(X) = log P(A) + log P(B) + log P(C)
    manual_ll = (
        mix_A.log_prob(test_data)
        + mix_B.log_prob(test_data)
        + mix_C.log_prob(test_data)
    )
    assert torch.allclose(log_p, manual_ll, atol=1e-5), "Product property violated"

    print("✓ Test 9.1 passed: Algorithm 1 construction pattern")


def test_overlap_inference_same_as_disjoint():
    """
    Test 9.2: With correct construction, inference is same as ProductOverGroups.

    Key Insight: ProductOverGroupsWithOverlap with proper construction
                 is functionally equivalent to ProductOverGroups.
    """
    np.random.seed(60)
    torch.manual_seed(60)

    X = np.random.randn(80, 5)

    # Create SPNs for disjoint groups
    spn0_g0 = LocalSPNWrapper(num_features=3, device="cpu", depth=1, seed=1)
    spn0_g1 = LocalSPNWrapper(num_features=2, device="cpu", depth=1, seed=2)

    spn0_g0.train_local(X[:, [0, 1, 2]], epochs=10)
    spn0_g1.train_local(X[:, [3, 4]], epochs=10)

    mix_g0 = GroupMixture([spn0_g0], weights=[1.0], feature_indices=[0, 1, 2])
    mix_g1 = GroupMixture([spn0_g1], weights=[1.0], feature_indices=[3, 4])

    # Create both product types
    product_disjoint = ProductOverGroups([mix_g0, mix_g1], [[0, 1, 2], [3, 4]])
    product_overlap = ProductOverGroupsWithOverlap(
        [mix_g0, mix_g1], [[0, 1, 2], [3, 4]], allow_overlap=True
    )

    # Verify no overlaps detected
    assert not product_overlap.overlap_info["has_overlap"], "Should have no overlaps"

    # Test inference: should be identical
    test_data = torch.tensor(X[:15], dtype=torch.float32)
    ll_disjoint = product_disjoint.log_prob(test_data)
    ll_overlap = product_overlap.log_prob(test_data)

    assert torch.allclose(
        ll_disjoint, ll_overlap, atol=1e-6
    ), "With proper construction (no overlaps), both classes should be equivalent"

    # Test sampling: shapes should match
    samples_disjoint = product_disjoint.sample(50)
    samples_overlap = product_overlap.sample(50)

    assert samples_disjoint.shape == samples_overlap.shape, "Sample shapes should match"
    assert samples_overlap.shape == (50, 5), "Sample shape mismatch"

    print("✓ Test 9.2 passed: Inference same as disjoint when properly constructed")


# ============================================================
# Run all tests
# ============================================================


if __name__ == "__main__":
    print("=" * 70)
    print("HYBRID CLASSES TEST SUITE (Days 1-4)")
    print("=" * 70)

    # Test 1-4: GroupMixture (Day 1)
    print("\n" + "=" * 70)
    print("GROUPMIXTURE TESTS (Day 1)")
    print("=" * 70)

    print("\n[Test 1] Initialization")
    test_groupmixture_init_basic()
    test_groupmixture_init_validation()

    print("\n[Test 2] log_prob computation")
    test_groupmixture_log_prob_identical_spns()
    test_groupmixture_log_prob_feature_extraction()

    print("\n[Test 3] Sampling")
    test_groupmixture_sample_shape()
    test_groupmixture_sample_distribution()

    print("\n[Test 4] Edge cases")
    test_groupmixture_univariate()

    # Test 5-7: ProductOverGroups (Day 3-4)
    print("\n" + "=" * 70)
    print("PRODUCTOVERGROUPS TESTS (Day 3-4)")
    print("=" * 70)

    print("\n[Test 5] Initialization")
    test_product_init_basic()
    test_product_overlap_detection()

    print("\n[Test 6] log_prob computation")
    test_product_log_prob_sum()
    test_product_log_prob_independence()

    print("\n[Test 7] Sampling")
    test_product_sample_shape()
    test_product_sample_independence()

    # Test 8-9: ProductOverGroupsWithOverlap (Day 5)
    print("\n" + "=" * 70)
    print("PRODUCTOVERGROUPSWITHOVERLAP TESTS (Day 5)")
    print("=" * 70)

    print("\n[Test 8] Overlap detection")
    test_overlap_detection_basic()
    test_overlap_detection_allowed()

    print("\n[Test 9] Algorithm 1 construction pattern")
    test_overlap_algorithm1_pattern()
    test_overlap_inference_same_as_disjoint()

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED (Days 1-5)!")
    print("=" * 70)
