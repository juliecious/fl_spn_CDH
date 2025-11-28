import logging
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from causallearn.utils.data_utils import (
    get_dag_from_pdag,
    count_skeleton_accuracy,
    count_dag_accuracy,
)
from causallearn.search.ConstraintBased.CDNOD import cdnod
from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset
from typing import List

from test_data import X_proc as X, true_DAG_bin, d, K

np.random.seed(42)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class SPNConditionalIndependenceTest:
    """SPN-based conditional independence test compatible with causallearn."""

    def __init__(self, model, data, device="cpu"):
        self.model = model
        self.data = (
            torch.tensor(data, dtype=torch.float32)
            if isinstance(data, np.ndarray)
            else data
        ).to(device)
        self.device = device

    def __call__(
        self, x_idx: int, y_idx: int, z_indices: List[int], threshold=1e-2
    ) -> float:
        """
        Calculates Conditional Mutual Information - CMI score.
        If score < threshold, variables are Independent.
        """
        cmi = self._calculate_cmi(x_idx, y_idx, z_indices)
        return cmi

    def _get_log_prob(self, current_vars: List[int]):
        """
        Helper to compute log prob of specific variables, marginalizing others out.
        current_vars: list of indices to KEEP. All others are set to NaN.
        """
        # Create a mask of NaNs
        batch_input = torch.full_like(self.data, float("nan"))

        # Fill in only the variables we are interested in
        for idx in current_vars:
            batch_input[:, idx] = self.data[:, idx]

        with torch.no_grad():
            self.model.eval()
            # simple_einet handles NaNs as marginalization
            return self.model(batch_input).mean().item()

    def _calculate_cmi(self, x_idx, y_idx, z_indices):
        """
        Correct Formula:
        CMI = log P(X,Y,Z) + log P(Z) - log P(X,Z) - log P(Y,Z)
        """
        # 1. log P(X, Y, Z)
        xyz_vars = [x_idx, y_idx] + z_indices
        log_p_xyz = self._get_log_prob(xyz_vars)

        # 2. log P(X, Z)
        xz_vars = [x_idx] + z_indices
        log_p_xz = self._get_log_prob(xz_vars)

        # 3. log P(Y, Z)
        yz_vars = [y_idx] + z_indices
        log_p_yz = self._get_log_prob(yz_vars)

        # 4. log P(Z)
        if not z_indices:
            log_p_z = 0.0
        else:
            log_p_z = self._get_log_prob(z_indices)

        # Calculate CMI
        cmi = log_p_xyz - log_p_xz - log_p_yz + log_p_z

        return max(0.0, cmi)  # Clamp to 0 to handle numerical noise


if __name__ == "__main__":
    # ===== TRAINING PHASE =====
    num_features = X.shape[1]
    config = EinetConfig(
        num_features=num_features,
        num_channels=1,
        num_sums=5,
        num_leaves=10,
        num_repetitions=5,
        num_classes=1,
        depth=3,  # int(np.ceil(np.log2(num_features))),
        dropout=0.0,
        leaf_type=Normal,
        layer_type="linsum",
        structure="top-down",
    )
    model = Einet(config)

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    logging.info(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(train_tensor), batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    n_epochs = 50

    logging.info("Starting Training...")
    for epoch in range(n_epochs):
        model.train()
        loss = 0.0
        for (batch_data,) in dataloader:
            optimizer.zero_grad()
            loss = -model(batch_data).mean()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            logging.info(f"[RAT-SPN] Epoch {epoch} Loss {loss.item():.4f}")

    spn_tester = SPNConditionalIndependenceTest(model, X_test)

    def spn_ci_wrapper(data, i, j, k, **kwargs):
        """
        The Oracle Function.
        i, j: Indices of variables X and Y
        k: List of indices for conditioning set Z
        """
        # Calculate CMI using your trained SPN
        # Note: 'k' might now include the domain index (c_indx) passed by CDNOD
        cmi_val = spn_tester._calculate_cmi(x_idx=i, y_idx=j, z_indices=list(k))

        # Thresholding (Tuning this is critical for SPNs)
        # If CMI is very low (e.g. < 0.02), we assume Independence.
        if cmi_val < 0.02:
            return 1.0  # Independent
        else:
            return 0.0  # Dependent

    c_indx = -1  # Usually the last column indicates the 'Context' or 'Client ID'

    logging.info("Running CDNOD with SPN Oracle...")

    # 3. Run CDNOD
    cg = cdnod(
        data=X_test,
        c_indx=c_indx,
        K=K,
        alpha=0.05,
        indep_test=spn_ci_wrapper,  # <--- Your SPN goes here
        uc_priority=-1,
        stable=True,
        verbose=True,
    )
    # # ===== TESTING PHASE =====
    # spn_tester = SPNConditionalIndependenceTest(model, X_test)
    #
    # CMI_THRESHOLD = 0.02
    # logging.info("-" * 40)
    # dependency_matrix = np.zeros((d, d))
    #
    # for i in range(d):
    #     for j in range(i + 1, d):
    #         # Unconditional check (Change z_indices to test conditional)
    #         cmi = spn_tester._calculate_cmi(x_idx=i, y_idx=j, z_indices=[])
    #
    #         is_dependent = 1 if cmi > CMI_THRESHOLD else 0
    #         dependency_matrix[i, j] = is_dependent
    #         dependency_matrix[j, i] = is_dependent
    #
    #         # Log only interesting ones to reduce clutter
    #         if is_dependent:
    #             status = "DEP"
    #         else:
    #             status = "IND"
    #
    #         logging.info(f"X{i}-X{j}: CMI={cmi:.4f} -> {status}")
    #
    # logging.info("-" * 40)
    # logging.info(f"Estimated Matrix:\n{dependency_matrix}")
    # est_dag_from_pdag = get_dag_from_pdag(dependency_matrix)
    # logging.info(f"Estimated DAG:\n{est_dag_from_pdag}")
    # logging.info(f"True Matrix:\n{true_DAG_bin}")
    #
    # # Undirected skeleton: F1, recall, precision, SHD
    # ret_skeleton = count_skeleton_accuracy(true_DAG_bin, dependency_matrix)
    # logging.info(f"Undirected skeleton:\n{ret_skeleton}")
    # # Directed graph: F1, recall, precision, SHD
    # ret_diretion = count_dag_accuracy(true_DAG_bin, est_dag_from_pdag)
    # logging.info(f"Directed graph:\n{ret_diretion}")
