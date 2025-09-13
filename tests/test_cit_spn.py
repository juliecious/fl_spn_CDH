import numpy as np
import torch

from causallearn.utils.cit import CIT, SPN

np.random.seed(42)


def test_spn_cit_basic():
    """
    Basic functionality test: can SPN CI be correctly initialized and compute unconditional/conditional CI tests?
    """
    print("SPN CI basic test passed!")

    # Generate reproducible random toy data (n samples, d dimensions)
    n, d = 100, 4
    X = np.random.randn(n, d)

    # Initialize SPN test object
    spn = SPN(data=X)

    # Initialize SPN CI test object, specify SPN-related parameters explicitly
    spn_ci = CIT(data=X, method="spn", epochs=10, lr=0.01, depth=2, num_leaves=6)

    # Unconditional independence test
    pvalue1 = spn_ci(0, 1, [])
    print("SPN unconditional CI p-value:", pvalue1)
    assert 0.0 <= pvalue1 <= 1.0
    #
    # Conditional independence test (conditioning on variable 2)
    pvalue2 = spn_ci(0, 1, [2])
    print("SPN CI (condition on 2) p-value:", pvalue2)
    assert 0.0 <= pvalue2 <= 1.0
    #
    # Multiple conditioning variables
    pvalue3 = spn_ci(0, 1, [2, 3])
    print("SPN CI (condition on 2,3) p-value:", pvalue3)
    assert 0.0 <= pvalue3 <= 1.0

    print("SPN CI basic test passed!")
    print("=" * 60)


def test_spn_pipeline_integrity():

    print("Testing SPN pipeline")
    X = np.random.randn(120, 4)
    spn_kwargs = dict(
        epochs=5,
        depth=2,
        num_sums=4,
        num_leaves=4,
        num_repetitions=2,
        dropout=0.0,
        lr=0.001,
    )
    # ========== SPN Init ==========
    spn = SPN(X, **spn_kwargs)
    assert hasattr(spn, "einet"), "SPN should have attribute 'einet'."
    assert hasattr(spn, "scaler"), "SPN should have attribute 'scaler'."
    # ========== Unconditional CI ==========
    p_uncond = spn(0, 1, [])
    print("Unconditional p-value:", p_uncond)
    assert 0.0 <= p_uncond <= 1.0
    # ========== Single conditional variable ==========
    p_cond1 = spn(0, 1, [2])
    print("Condition on [2] p-value:", p_cond1)
    assert 0.0 <= p_cond1 <= 1.0
    # ========== Multi-conditional var ==========
    p_cond2 = spn(0, 1, [2, 3])
    print("Condition on [2,3] p-value:", p_cond2)
    assert 0.0 <= p_cond2 <= 1.0

    # ========== Test cache with multiple calls ==========
    p_uncond2 = spn(0, 1, [])
    assert (
        p_uncond2 == p_uncond
    ), "p-value caching failed, repeated call with same args should return same value."

    # ========== Test invalid index ==========
    try:
        spn(0, 5, [])  # 超出維度
        raised = False
    except Exception:
        raised = True
    assert raised, "Should raise error on invalid index!"

    # ========== Multi-type arguments support ==========
    p_alt = spn([0], [1], [2])
    assert 0.0 <= p_alt <= 1.0

    # ========== 輸出皆應非NaN/非Inf ==========
    assert not np.isnan(p_uncond)
    assert not np.isnan(p_cond1)
    assert not np.isnan(p_cond2)
    assert not np.isinf(p_uncond)
    assert not np.isinf(p_cond1)
    assert not np.isinf(p_cond2)

    print("SPN pipeline all tests passed!")
    print("=" * 60)


def test_spn_pvalue_variety():
    print("Testing SPN p-value variety")

    n, d = 200, 4
    X = np.random.randn(n, d)
    # add artificial noise X[:,1] = X[:,0] + noise
    X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n)

    spn_kwargs = dict(
        epochs=5,
        depth=2,
        num_sums=4,
        num_leaves=4,
        num_repetitions=2,
        dropout=0.0,
        lr=0.001,
    )
    spn = SPN(X, **spn_kwargs)

    # Unconditional p-value
    p1 = spn(0, 1, [])  # X0 vs X1, should be dependent in this synthetic data
    p2 = spn(0, 2, [])  # X0 vs X2, should be nearly independent (if X2 is just noise)
    p3 = spn(0, 1, [2])  # Conditioned on X2
    p4 = spn(0, 1, [3])  # Conditioned on X3

    print(f"p(0,_1|[]) = {p1:.4f}")
    print(f"p(0,_2|[]) = {p2:.4f}")
    print(f"p(0,_1|[2]) = {p3:.4f}")
    print(f"p(0,_1|[3]) = {p4:.4f}")

    assert 0.0 <= p1 <= 1.0
    assert 0.0 <= p2 <= 1.0
    assert 0.0 <= p3 <= 1.0
    assert 0.0 <= p4 <= 1.0

    assert (
        abs(p1 - p2) > 0.02
    ), "p-value should be different when the correlation structure is different"
    assert (
        abs(p1 - p3) > 1e-4 or abs(p1 - p4) > 1e-4
    ), "Conditioning should affect p-value (unless truly independent)"

    print("SPN p-value variety test passed.")
    print("=" * 60)


def test_cit_spn_pipeline():
    # Synthetic test data
    n, d = 120, 4
    X = np.random.randn(n, d)
    # Design correlation: let variable 1 almost copy variable 0
    X[:, 1] = X[:, 0] + np.random.randn(n) * 0.05

    spn_kwargs = dict(
        method="spn",
        epochs=5,
        depth=2,
        num_sums=4,
        num_leaves=4,
        num_repetitions=2,
        dropout=0.0,
        lr=0.001,
    )

    # --- Use CIT factory to create the SPN object ---
    # This is the standard pipeline in your repo!
    print("Testing CIT(spn)...")
    ci_test = CIT(X, **spn_kwargs)

    # --- Test unconditional independence ---
    p01 = ci_test(0, 1, [])
    p02 = ci_test(0, 2, [])
    p13 = ci_test(1, 3, [])
    # Conditional independence
    p01_c2 = ci_test(0, 1, [2])
    p01_c3 = ci_test([0], [1], [3])  # List interface (full pipeline check)
    p20 = ci_test([2], [0], [])

    print(f"CIT(SPN) p(0,1|[]) = {p01:.4f}")
    print(f"CIT(SPN) p(0,2|[]) = {p02:.4f}")
    print(f"CIT(SPN) p(1,3|[]) = {p13:.4f}")
    print(f"CIT(SPN) p(0,1|[2]) = {p01_c2:.4f}")
    print(f"CIT(SPN) p([0],[1]|[3]) = {p01_c3:.4f}")
    print(f"CIT(SPN) p([2],[0]|[]) = {p20:.4f}")

    # Results should all be within [0,1]
    for p in [p01, p02, p13, p01_c2, p01_c3, p20]:
        assert 0.0 <= p <= 1.0, "p-value out of valid range!"
    # p(0,1|[]) (dependent) should differ from p(0,2|[]) (independent)
    assert abs(p01 - p02) > 0.01, "SPN p-values did not react to true data structure!"
    print("CIT(SPN) pipeline test passed.")
    print("=" * 60)


def test_cit_all_methods():
    # Generate synthetic data with dependence and independence
    np.random.seed(42)
    n, d = 120, 4
    X = np.random.randn(n, d)
    # Make variable 1 strongly correlated with variable 0
    X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n)
    # Variable 2 is independent

    methods = [
        dict(method="kci"),  # Kernel-based
        dict(method="gsq"),  # G-squared (categorical)
        dict(
            method="spn",
            epochs=5,
            depth=2,
            num_sums=4,
            num_leaves=4,
            num_repetitions=2,
            dropout=0.0,
            lr=0.001,
        ),  # SPN config
    ]
    method_names = ["KCI", "GSQ", "SPN"]
    X_gsq = (X * 2).astype(int)
    results = {}

    for method, name in zip(methods, method_names):
        print(f"\nTesting method: {name}")
        data = X_gsq if name == "GSQ" else X
        ci_test = CIT(data, **method)
        res = {}
        # --- For old backend: use int,int,[list], for SPN: also check list/list/list
        res["p(0,1|[])"] = ci_test(0, 1, [])
        res["p(0,2|[])"] = ci_test(0, 2, [])
        res["p(0,1|[2])"] = ci_test(0, 1, [2])
        if name == "SPN":
            res["p([0],[1]|[3])"] = ci_test([0], [1], [3])
        else:
            # For old-style, stick to ints otherwise might crash
            res["p(1,3|[2])"] = ci_test(1, 3, [2])
        for key, val in res.items():
            print(f"{name} {key} = {val:.4f}")
            assert 0.0 <= val <= 1.0
        results[name] = res

    # Ensures methods are not all always identical (if so, bug in dispatch)
    vals = [tuple(i.values()) for i in results.values()]
    assert any(
        vals[0] != v for v in vals[1:]
    ), "All methods returned identical p-values? Check method dispatch logic!"

    print("\nMulti-method CIT pipeline (KCI, GSQ, SPN) test passed.")


if __name__ == "__main__":
    # test_spn_cit_basic()
    # test_spn_pipeline_integrity()
    # test_spn_pvalue_variety()
    # test_cit_spn_pipeline()
    test_cit_all_methods()
