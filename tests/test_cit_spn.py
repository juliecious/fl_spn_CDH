import numpy as np
import torch

from causallearn.utils.cit import CIT, SPN


def test_spn_cit_basic():
    """
    Basic functionality test: can SPN CI be correctly initialized and compute unconditional/conditional CI tests?
    """
    # Generate reproducible random toy data (n samples, d dimensions)
    np.random.seed(0)
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


if __name__ == "__main__":
    test_spn_cit_basic()
