import logging
import numpy as np
import torch
from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset
from typing import List

from test_data import X, true_DAG_bin

np.random.seed(42)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class SPNConditionalIndependenceTest:
    """SPN-based conditional independence test compatible with causallearn."""

    def __init__(self, model, data, n_permutations=100):
        self.model = model
        self.data = (
            torch.tensor(data, dtype=torch.float32)
            if isinstance(data, np.ndarray)
            else data
        )
        self.n_permutations = n_permutations

    def __call__(self, data, x_idx: int, y_idx: int, z_indices: List[int]) -> float:
        """Test: X ⊥ Y | Z"""
        return self._calculate_p_value(x_idx, y_idx, z_indices)

    def _calculate_mutual_info(self, x_idx, y_idx, z_indices):
        """Calculate: I(X; Y | Z)"""
        n_samples = self.data.shape[0]
        all_nan = torch.full((n_samples, self.data.shape[1]), float("nan"))

        # P(X, Y | Z)
        input_xy_z = all_nan.clone()
        input_xy_z[:, x_idx] = self.data[:, x_idx]
        input_xy_z[:, y_idx] = self.data[:, y_idx]
        for z_idx in z_indices:
            input_xy_z[:, z_idx] = self.data[:, z_idx]
        log_p_xy_z = self.model(input_xy_z).mean().item()

        # P(X | Z)
        input_x_z = all_nan.clone()
        input_x_z[:, x_idx] = self.data[:, x_idx]
        for z_idx in z_indices:
            input_x_z[:, z_idx] = self.data[:, z_idx]
        log_p_x_z = self.model(input_x_z).mean().item()

        # P(Y | Z)
        input_y_z = all_nan.clone()
        input_y_z[:, y_idx] = self.data[:, y_idx]
        for z_idx in z_indices:
            input_y_z[:, z_idx] = self.data[:, z_idx]
        log_p_y_z = self.model(input_y_z).mean().item()

        # Conditional mutual information
        cond_mutual_info = log_p_xy_z - log_p_x_z - log_p_y_z
        return cond_mutual_info

    def _calculate_p_value(self, x_idx, y_idx, z_indices):
        """Monte Carlo permutation test for conditional independence."""
        observed_mi = self._calculate_mutual_info(x_idx, y_idx, z_indices)
        permuted_mis = []

        data_np = (
            self.data.numpy() if isinstance(self.data, torch.Tensor) else self.data
        )

        for _ in range(self.n_permutations):
            # Shuffle X while keeping Y and Z fixed
            permuted_data = torch.tensor(data_np.copy(), dtype=torch.float32)
            permuted_indices = np.random.permutation(data_np.shape[0])
            permuted_data[:, x_idx] = permuted_data[permuted_indices, x_idx]

            # Temporarily swap data
            original_data = self.data
            self.data = permuted_data

            permuted_mi = self._calculate_mutual_info(x_idx, y_idx, z_indices)
            permuted_mis.append(permuted_mi)

            # Restore original data
            self.data = original_data

        # Calculate empirical p-value
        p_value = np.mean(np.array(permuted_mis) >= observed_mi)
        return max(p_value, 1.0 / self.n_permutations)


if __name__ == "__main__":
    # ===== TRAINING PHASE =====
    num_features = X.shape[1]
    depth = int(np.ceil(np.log2(num_features)))
    config = EinetConfig(
        num_features=num_features,
        num_channels=1,
        num_sums=5,
        num_leaves=8,
        num_repetitions=4,
        num_classes=1,
        depth=depth,
        dropout=0.0,
        leaf_type=Normal,
        layer_type="linsum",
        structure="top-down",
    )
    model = Einet(config)

    # Use all data for training (no holdout)
    train_tensor = torch.tensor(X, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(train_tensor), batch_size=100, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5
    )
    best_ll = float("-inf")
    patience, max_patience = 0, 30
    n_epochs = 300

    # Training Loop
    for epoch in range(n_epochs):
        model.train()
        epoch_loss, epoch_ll = 0, 0
        for (batch_data,) in dataloader:
            optimizer.zero_grad()
            log_lls = model(batch_data)
            loss = -log_lls.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

    # Evaluation
    model.eval()
    with torch.no_grad():
        full_log_ll = model(train_tensor).mean().item()
    logging.info(f"[EiNet] Full dataset average log-likelihood: {full_log_ll:.4f}")

    # ===== TESTING PHASE =====
    logging.info("\n" + "=" * 60)
    logging.info("TESTING SPN CONDITIONAL INDEPENDENCE TEST")
    logging.info("=" * 60)

    # Initialize the CI test
    spn_ci_test = SPNConditionalIndependenceTest(model, X, n_permutations=1000)
    p_val_threshold = 0.01

    # Test 1: Unconditional independence (empty conditioning set)
    logging.info("\n[Test 1] Unconditional Independence: X_0 ⊥ X_1")
    p_value_01 = spn_ci_test(X, x_idx=0, y_idx=1, z_indices=[])
    logging.info(f"  p-value: {p_value_01:.4f}")
    logging.info(
        f"  Decision (α=0.05): {'Independent' if p_value_01 > p_val_threshold else 'Dependent'}"
    )

    # Test 2: Unconditional independence (different pair)
    logging.info("\n[Test 2] Unconditional Independence: X_0 ⊥ X_2")
    p_value_02 = spn_ci_test(X, x_idx=0, y_idx=2, z_indices=[])
    logging.info(f"  p-value: {p_value_02:.4f}")
    logging.info(
        f"  Decision (α=0.05): {'Independent' if p_value_02 > p_val_threshold else 'Dependent'}"
    )

    # Test 3: Conditional independence with single conditioning variable
    logging.info("\n[Test 3] Conditional Independence: X_0 ⊥ X_1 | X_2")
    p_value_01_c2 = spn_ci_test(X, x_idx=0, y_idx=1, z_indices=[2])
    logging.info(f"  p-value: {p_value_01_c2:.4f}")
    logging.info(
        f"  Decision (α=0.05): {'Independent' if p_value_01_c2 > p_val_threshold else 'Dependent'}"
    )

    # Test 4: Conditional independence with multiple conditioning variables
    logging.info("\n[Test 4] Conditional Independence: X_0 ⊥ X_1 | X_2, X_3")
    p_value_01_c23 = spn_ci_test(X, x_idx=0, y_idx=1, z_indices=[2, 3])
    logging.info(f"  p-value: {p_value_01_c23:.4f}")
    logging.info(
        f"  Decision (α=0.05): {'Independent' if p_value_01_c23 > p_val_threshold else 'Dependent'}"
    )

    # Test 5: Compare with true DAG (if available)
    logging.info("\n[Test 5] Comparison with True DAG")
    logging.info(f"  True DAG:\n{true_DAG_bin}")

    # Test all pairs systematically
    logging.info("\n[Test 6] Systematic Pairwise Testing")
    n_features = X.shape[1]
    dependency_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i + 1, n_features):
            p_value = spn_ci_test(X, x_idx=i, y_idx=j, z_indices=[])
            is_dependent = 1 if p_value < p_val_threshold else 0
            dependency_matrix[i, j] = is_dependent
            dependency_matrix[j, i] = is_dependent
            logging.info(
                f"  X_{i} ⊥ X_{j}: p-value={p_value:.4f}, Dependent={is_dependent}"
            )

    logging.info(f"\n  Estimated Dependency Matrix:\n{dependency_matrix}")
    logging.info("=" * 60)
