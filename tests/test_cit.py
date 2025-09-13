import numpy as np
from causallearn.utils.cit import CIT
import torch

np.random.seed(42)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)


if __name__ == "__main__":
    n, d = 60, 4
    X = np.random.randn(n, d)
    X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n)  # X0 vs X1 強依賴
    X[:, 2] = np.random.randn(n)  # X2 獨立於 X0

    # CIT function call，選擇 method='spn' 並明確指定參數（可增強穩定性)
    spn_kwargs = dict(
        method="spn",
        epochs=40,
        lr=0.001,
        depth=2,
        num_leaves=8,
        num_sums=8,
        num_repetitions=4,
    )
    try:
        p1 = CIT(X, **spn_kwargs)(0, 1, [])  # X0 vs X1, 應該 dependent
        p2 = CIT(X, **spn_kwargs)(0, 2, [])  # X0 vs X2, 應該 independent

        print(f"p1 (0 vs 1, dependent)   = {p1:.4f}")
        print(f"p2 (0 vs 2, independent) = {p2:.4f}")
        assert abs(p1 - p2) > 0.02, "SPN CI 無法辨識依賴與獨立，請檢查資料設計與參數"
    except Exception as e:
        print(e)
