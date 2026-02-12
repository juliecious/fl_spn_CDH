import time
import numpy as np
import pandas as pd
import torch
import logging
from scipy.stats import spearmanr
import sys
import os

# 設定日誌級別
logging.basicConfig(level=logging.INFO, format="%(message)s")

# 確保可以導入你的模組
sys.path.append(os.getcwd())

try:
    from causallearn.utils.cit import CIT, SPN_CIT
    from causallearn.utils.FedPC import GlobalFedSPN, LocalSPNWrapper
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("請確保你在 'fl_spn_CDH' 目錄下運行此腳本，且 causallearn 包在路徑中。")
    sys.exit(1)

# 強制使用 CPU 進行公平對比
DEVICE = "cpu"
print(f"Running Benchmark on: {DEVICE}")


def generate_synthetic_data(n_samples, scenario):
    """
    生成三種典型場景的合成數據 (X, Y, Z)
    """
    np.random.seed(42)
    # Z 是混淆因子 (Confounder) 或 父節點
    Z = np.random.uniform(-2, 2, n_samples)

    if scenario == "Linear_Dependent":
        # 線性依賴: X <- Z -> Y, 且 X -> Y (Direct Edge)
        # X 與 Y 給定 Z 仍相關
        X = 0.5 * Z + np.random.normal(0, 0.5, n_samples)
        Y = 0.5 * Z + 0.5 * X + np.random.normal(0, 0.5, n_samples)
        is_independent = False

    elif scenario == "NonLinear_Dependent":
        # 非線性依賴: Y = cos(X) + Z
        # KCI 應該能檢測出來，線性 FisherZ 會失敗
        X = np.random.uniform(-2, 2, n_samples)
        Y = np.cos(X) + Z + np.random.normal(0, 0.1, n_samples)
        is_independent = False

    elif scenario == "NonLinear_Independent":
        # 條件獨立: X <- Z -> Y (無 X->Y 邊)
        # X = Z^2, Y = tanh(Z)
        # X _||_ Y | Z 應該成立
        X = Z**2 + np.random.normal(0, 0.2, n_samples)
        Y = np.tanh(Z) + np.random.normal(0, 0.2, n_samples)
        is_independent = True

    else:
        raise ValueError("Unknown scenario")

    data = np.column_stack((X, Y, Z))
    return data, is_independent


def train_fedspn_oracle(data):
    """
    快速訓練一個 GlobalFedSPN 作為 CI Test 的 Oracle。
    這裡模擬了聯邦學習的聚合結果。
    """
    n_samples, n_features = data.shape

    # 簡單模擬：將數據分為 2 個 Cluster (Mixture) 來擬合非線性
    # 這對應於 FedCDH 中的 SimulatedFederatedKMeans + Vertical/Horizontal Split
    from sklearn.cluster import KMeans

    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
    labels = kmeans.labels_

    components = []
    weights = []

    for c in range(n_clusters):
        cluster_data = data[labels == c]
        if len(cluster_data) < 10:
            continue

        # 訓練局部 SPN
        # 注意：這是 "Training Cost"，在 PC 算法中只發生一次
        leaf = LocalSPNWrapper(
            num_features=n_features,
            device=DEVICE,
            num_sums=20,  # 增加模型容量
            num_leaves=10,
            num_repetitions=5,  # 增加模型容量
            depth=1,
        )
        # 使用 50 epochs 確保收斂 (根據你的最新修正)
        leaf.train_local(cluster_data, epochs=50, lr=0.01)

        components.append(leaf)
        weights.append(len(cluster_data) / n_samples)

    global_spn = GlobalFedSPN(components, weights=weights, device=DEVICE)
    return global_spn


def run_benchmark(n_values=[200, 500]):
    results = []
    scenarios = ["Linear_Dependent", "NonLinear_Dependent", "NonLinear_Independent"]

    # 預熱 (Warmup)
    print("Warming up models...")
    _ = generate_synthetic_data(100, "Linear_Dependent")

    for n in n_values:
        print(f"\n{'='*20} Sample Size N = {n} {'='*20}")

        for scenario in scenarios:
            data, gt_indep = generate_synthetic_data(n, scenario)
            print(
                f"\nScenario: {scenario} (Ground Truth: {'Independent' if gt_indep else 'Dependent'})"
            )

            # --- 1. FedSPN (Proposed) ---
            t0 = time.time()
            global_spn = train_fedspn_oracle(data)
            train_time = time.time() - t0

            # 初始化 Tester (使用你剛修復的 Permutation Test)
            # 為了測試速度，我們設 num_permutations=100 (足夠穩定且能低於 0.01)
            spn_tester = SPN_CIT(data, global_model=global_spn, num_permutations=100)

            t1 = time.time()
            # Test: X(0) _||_ Y(1) | Z(2)
            p_spn = spn_tester(0, 1, [2])
            spn_infer_time = time.time() - t1

            spn_pred_indep = p_spn > 0.05  # Alpha = 0.05
            spn_correct = spn_pred_indep == gt_indep

            # --- 2. KCI (Baseline) ---
            # KCI 沒有顯式的訓練階段 (Lazy Learning)，計算都在 Inference
            kci_tester = CIT(data, "kci")

            t2 = time.time()
            p_kci = kci_tester(0, 1, [2])
            kci_infer_time = time.time() - t2

            kci_pred_indep = p_kci > 0.05
            kci_correct = kci_pred_indep == gt_indep

            # --- 記錄結果 ---
            print(
                f"  [FedSPN] Train: {train_time:.3f}s | Infer: {spn_infer_time:.4f}s | P-val: {p_spn:.3f} | Correct: {spn_correct}"
            )
            print(
                f"  [KCI   ] Train: 0.000s | Infer: {kci_infer_time:.4f}s | P-val: {p_kci:.3f} | Correct: {kci_correct}"
            )

            results.append(
                {
                    "N": n,
                    "Scenario": scenario,
                    "Method": "FedSPN",
                    "Train_Time": train_time,
                    "Infer_Time": spn_infer_time,
                    "Total_Time": train_time + spn_infer_time,
                    "Correct": spn_correct,
                }
            )
            results.append(
                {
                    "N": n,
                    "Scenario": scenario,
                    "Method": "KCI",
                    "Train_Time": 0,
                    "Infer_Time": kci_infer_time,
                    "Total_Time": kci_infer_time,
                    "Correct": kci_correct,
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_benchmark()

    print("\n\n" + "=" * 50)
    print("BENCHMARK SUMMARY (Inference Speedup)")
    print("=" * 50)

    # 透視表：比較推理時間
    pivot_infer = df.pivot_table(
        index=["N", "Scenario"], columns="Method", values="Infer_Time"
    )
    pivot_infer["Speedup (KCI/SPN)"] = pivot_infer["KCI"] / pivot_infer["FedSPN"]
    print(pivot_infer)

    print("\n\n" + "=" * 50)
    print("BENCHMARK SUMMARY (Accuracy)")
    print("=" * 50)
    pivot_acc = df.pivot_table(index=["Scenario"], columns="Method", values="Correct")
    print(pivot_acc)
