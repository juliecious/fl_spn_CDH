import numpy as np
import torch
from causallearn.utils.cit import CIT, SPN


def test_spn_cit_basic():
    """
    基本功能測試：保證 SPN 可以初始化並正確執行無條件及有條件獨立性檢定。
    """
    np.random.seed(0)
    n, d = 100, 4
    X = np.random.randn(n, d)

    # 明確指定參數初始化 SPN 實例
    spn = SPN(data=X, epochs=10, lr=0.01, depth=2, num_leaves=6)
    # 用 CIT 包裝呼叫亦可
    spn_ci = CIT(data=X, method="spn", epochs=10, lr=0.01, depth=2, num_leaves=6)

    # 無條件獨立性
    pvalue1 = spn_ci(0, 1, [])
    print("Unconditional CI p-value", pvalue1)
    assert 0.0 <= pvalue1 <= 1.0

    # 有條件獨立性 (cond on 2)
    pvalue2 = spn_ci(0, 1, [2])
    print("Conditional CI (on 2) p-value", pvalue2)
    assert 0.0 <= pvalue2 <= 1.0

    # 多重條件變數
    pvalue3 = spn_ci(0, 1, [2, 3])
    print("Conditional CI (on 2,3) p-value", pvalue3)
    assert 0.0 <= pvalue3 <= 1.0

    print("SPN CI basic test passed.")
    print("=" * 60)


def test_spn_pipeline_integrity():
    """
    覆蓋 SPN pipeline 關鍵路徑，包括屬性驗證、p-value 是否一致性、支持多型態 index 格式及無效 index 處理。
    """
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
    spn = SPN(X, **spn_kwargs)

    assert hasattr(spn, "einet")
    assert hasattr(spn, "scaler")

    # 無條件獨立
    p_uncond = spn(0, 1, [])
    print("Unconditional p-value:", p_uncond)
    assert 0.0 <= p_uncond <= 1.0

    # 有條件
    p_cond1 = spn(0, 1, [2])
    assert 0.0 <= p_cond1 <= 1.0
    p_cond2 = spn(0, 1, [2, 3])
    assert 0.0 <= p_cond2 <= 1.0

    # 測試 cache 一致性
    p_uncond2 = spn(0, 1, [])
    assert p_uncond2 == p_uncond

    # 錯誤 index 處理
    raised = False
    try:
        spn(0, 5, [])
    except Exception:
        raised = True
    assert raised, "Should raise error on invalid index!"

    # 支持多種 index 類型傳遞
    p_alt = spn([0], [1], [2])
    assert 0.0 <= p_alt <= 1.0

    print("SPN pipeline all tests passed.")
    print("=" * 60)


def test_spn_pvalue_variety():
    """
    檢查 p-value 是否能如預期反映不同相關結構與條件。
    """
    n, d = 200, 4
    X = np.random.randn(n, d)
    X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n)  # dependent
    X[:, 2] = np.random.randn(n)  # independent

    spn_kwargs = dict(
        epochs=40,
        depth=2,
        num_sums=8,
        num_leaves=8,
        num_repetitions=4,
        dropout=0.0,
        lr=0.001,
    )
    spn = SPN(X, **spn_kwargs)

    p1 = spn(0, 1, [])
    p2 = spn(0, 2, [])

    print(f"p1 (X0 vs X1, dependent): {p1:.4f}")
    print(f"p2 (X0 vs X2, independent): {p2:.4f}")
    assert abs(p1 - p2) > 0.02

    print("SPN p-value variety test passed.")
    print("=" * 60)


if __name__ == "__main__":
    test_spn_cit_basic()
    test_spn_pipeline_integrity()
    test_spn_pvalue_variety()
