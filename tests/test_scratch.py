import numpy as np
import torch


def main():
    print("=" * 60)
    print("SPN CONDITIONAL INDEPENDENCE FULL PIPELINE TEST")
    print("=" * 60)
    from causallearn.utils.cit import SPN

    np.random.seed(42)
    n = 200

    # Generate a dataset with very strong dependence and independence structures
    X0 = np.random.randn(n)
    X1 = X0 + 0.01 * np.random.randn(n)  # X1 is almost a copy of X0 (highly dependent)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)
    X = np.column_stack([X0, X1, X2, X3])

    print(f"Correlation X0-X1: {np.corrcoef(X0, X1)[0,1]:.4f}")
    print(f"Correlation X0-X2: {np.corrcoef(X0, X2)[0,1]:.4f}")
    print("=" * 40)

    # Instantiate the SPN model (assuming your Bootstrap-based SPN implementation is active)
    spn = SPN(
        X, epochs=30, depth=2, num_sums=8, num_leaves=8, num_repetitions=4, lr=0.005
    )

    # --- Marginal log-likelihood and statistic computation ---
    data_tensor = spn.data_tensor[:80]
    ll_01 = spn._marginal_loglik(data_tensor, [0, 1]).mean().item()
    ll_0 = spn._marginal_loglik(data_tensor, [0]).mean().item()
    ll_1 = spn._marginal_loglik(data_tensor, [1]).mean().item()
    ll_02 = spn._marginal_loglik(data_tensor, [0, 2]).mean().item()
    ll_2 = spn._marginal_loglik(data_tensor, [2]).mean().item()

    stat_01 = ll_01 - ll_0 - ll_1
    stat_02 = ll_02 - ll_0 - ll_2

    print("Marginal Likelihoods:")
    print(f"  LL(X0,X1) = {ll_01:.4f}, LL(X0)={ll_0:.4f}, LL(X1)={ll_1:.4f}")
    print(f"  LL(X0,X2) = {ll_02:.4f}, LL(X2)={ll_2:.4f}")
    print(f"  Statistic (0,1): {stat_01:.5f}")
    print(f"  Statistic (0,2): {stat_02:.5f}")
    print("=" * 40)

    # --- Main CI test via SPN ---
    p_01 = spn(0, 1, [])
    p_02 = spn(0, 2, [])
    p_01_2 = spn(0, 1, [2])  # Conditioning variable provided
    p_01_3 = spn([0], [1], [3])  # Alternative argument type

    print("SPN p-values:")
    print(f"  p(0,1|[])     = {p_01:.4f}  (highly dependent, should be low)")
    print(f"  p(0,2|[])     = {p_02:.4f}  (independent, should be high)")
    print(f"  p(0,1|[2])    = {p_01_2:.4f}  (conditional test)")
    print(f"  p([0],[1]|[3])= {p_01_3:.4f}  (conditional test, alternative format)")
    print("=" * 40)

    # --- Bootstrap null distribution direct probe ---
    def bootstrap_probe(spn, xidx, yidx, n_bootstrap=40):
        # Original test statistic
        data_subset = spn.data_tensor[:80]
        ll_xy = spn._marginal_loglik(data_subset, xidx + yidx).mean().item()
        ll_x = spn._marginal_loglik(data_subset, xidx).mean().item()
        ll_y = spn._marginal_loglik(data_subset, yidx).mean().item()
        original_stat = ll_xy - ll_x - ll_y
        null_stats = []
        y_tensor_indices = torch.tensor(yidx, dtype=torch.long)
        for _ in range(n_bootstrap):
            shuffled_data = data_subset.clone()
            perm_indices = torch.randperm(shuffled_data.size(0))
            for y_col in y_tensor_indices:
                shuffled_data[:, y_col] = data_subset[perm_indices, y_col]
            ll_xy_null = spn._marginal_loglik(shuffled_data, xidx + yidx).mean().item()
            ll_x_null = spn._marginal_loglik(shuffled_data, xidx).mean().item()
            ll_y_null = spn._marginal_loglik(shuffled_data, yidx).mean().item()
            null_stat = ll_xy_null - ll_x_null - ll_y_null
            null_stats.append(null_stat)
        null_stats = np.array(null_stats)
        p_boot = np.mean(np.abs(null_stats) >= abs(original_stat))
        print(
            f"  Bootstrap (x={xidx},y={yidx}): orig={original_stat:.5f}, null mean={null_stats.mean():.5f}, std={null_stats.std():.5f}, p={p_boot:.4f}"
        )

    print("Bootstrap permutation probe:")
    bootstrap_probe(spn, [0], [1])
    bootstrap_probe(spn, [0], [2])
    print("=" * 40)

    # --- Unified test criteria ---
    # For dependent variables, p-value should be less than 0.05; for independent, greater than 0.3
    pass_all = (p_01 < 0.05) and (p_02 > 0.3)
    print(
        f"\n{'PASS' if pass_all else 'FAIL'}: SPN can accurately distinguish highly dependent from independent variable pairs"
    )


if __name__ == "__main__":
    main()
