import logging

from causallearn.utils.cit import fisherz
from simple_einet.einet import Einet, EinetConfig
import numpy as np
import torch
from scipy.stats import entropy
from simple_einet.layers.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset

from test_data import X, true_DAG_bin

np.random.seed(42)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


if __name__ == "__main__":
    num_features = X.shape[1]
    depth = int(np.ceil(np.log2(num_features)))

    config = EinetConfig(
        num_features=num_features,
        num_channels=1,
        num_sums=5,  # 每層 sum node 數
        num_leaves=8,  # 葉層分群數，可微調
        num_repetitions=4,  # 隨機區域分割重複次數（寬度）
        num_classes=1,
        depth=depth,  # 根據論文公式動態設置
        dropout=0.0,
        leaf_type=Normal,  # 連續資料建議選Normal
        layer_type="linsum",  # 預設
        structure="top-down",
    )
    spn_model = Einet(config)

    # Split train/test
    train_tensor = torch.tensor(X, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(train_tensor), batch_size=100, shuffle=True)

    # optimizer and scheduler setup
    optimizer = torch.optim.Adam(spn_model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5
    )
    best_ll = float("-inf")
    patience, max_patience = 0, 30
    n_epochs = 100

    # Training Loop
    for epoch in range(n_epochs):
        spn_model.train()
        epoch_loss, epoch_ll = 0, 0
        for (batch_data,) in dataloader:
            optimizer.zero_grad()
            log_lls = spn_model(batch_data)
            loss = -log_lls.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(spn_model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_ll += log_lls.mean().item()
        avg_ll = epoch_ll / len(dataloader)
        scheduler.step(-avg_ll)
        if epoch % 20 == 0:
            logging.info(
                f"[RAT-SPN] Epoch {epoch} Loss {(epoch_loss / len(dataloader)):.4f}  LogLL {avg_ll:.4f}  Patience {patience}"
            )
        if avg_ll > best_ll:
            best_ll = avg_ll
            patience = 0
        else:
            patience += 1
        if patience > max_patience:
            logging.info(f"[RAT-SPN] Early stopping at epoch {epoch}")
            break

    # Evaluation (log-likelihood on full dataset)
    spn_model.eval()
    with torch.no_grad():
        full_log_ll = spn_model(train_tensor).mean().item()
    logging.info(f"[EiNet] Full dataset average log-likelihood: {full_log_ll:.4f}")

    # Test independence on full dataset
    def calculate_dependency(model, data, idx1, idx2):
        "Calculate dependency from features."
        n_samples = data.shape[0]
        all_nan = torch.full((n_samples, data.shape[1]), float("nan"))
        # Convert data to torch tensor if it's a numpy array
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        # Joint log likelihood P(x1, x2)
        input_joint = all_nan.clone()
        input_joint[:, idx1] = data[:, idx1]
        input_joint[:, idx2] = data[:, idx2]
        log_p_joint = model(input_joint).mean().item()

        # Marginal log likelihood P(x1)
        input_x1 = all_nan.clone()
        input_x1[:, idx1] = data[:, idx1]
        log_p_x1 = model(input_x1).mean().item()

        # Marginal log likelihood P(x2)
        input_x2 = all_nan.clone()
        input_x2[:, idx2] = data[:, idx2]
        log_p_x2 = model(input_x2).mean().item()

        # Mutual information
        mutual_info = log_p_joint - log_p_x1 - log_p_x2
        return mutual_info

    def calculate_p_value(model, data, idx1, idx2, n_permutations=100):
        """Check if the result is robust with monte carlo method."""
        observed_mi = calculate_dependency(model, data, idx1, idx2)
        permuted_mis = []
        for _ in range(n_permutations):
            permuted_data = data.copy()
            np.random.shuffle(permuted_data[:, idx2])  # Shuffle one feature
            permuted_mi = calculate_dependency(model, permuted_data, idx1, idx2)
            permuted_mis.append(permuted_mi)
        p_value = np.mean(np.array(permuted_mis) >= observed_mi)
        return p_value

    # Example usage
    p = calculate_p_value(spn_model, X, 2, 4)
    print(f"Features p-value: {p}")
