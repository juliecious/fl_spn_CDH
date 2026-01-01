import numpy as np
import torch
import logging
from typing import List, Dict, Optional, Union
from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import optuna
from sklearn.model_selection import train_test_split
import copy


class FederatedSPNBase:
    """
    Base class containing shared utilities for Federated SPNs.
    """

    def __init__(self, device="cpu"):
        self.device = device


class ClientSPN(FederatedSPNBase):
    """
    Client Node: Trains a local SPN on its private data partition.
    Supports learning K components (clusters) for Vertical/Hybrid scenarios.
    """

    def __init__(
        self,
        client_id: int,
        num_features: int,
        num_clusters: int = 1,  # K in the paper (1 for Horizontal, >1 for Vertical)
        device="cpu",
        depth: int = None,  # If None, calculated automatically
        num_sums: int = 10,  # K in Sum-Product (Capacity)
        num_leaves: int = 20,  # Resolution of leaf distributions
        num_repetitions: int = 5,  # Ensembling factor
        lr: float = 0.05,
        batch_size: int = 32,
    ):
        super().__init__(device)
        self.client_id = client_id
        self.num_features = num_features
        self.num_clusters = num_clusters
        self.lr = lr
        self.batch_size = batch_size

        # Calculate the mathematical limit for this specific client's feature count
        # e.g., if features=6, log2(6)=2.58 -> max_depth=2
        physical_max_depth = int(np.floor(np.log2(num_features)))

        actual_depth = min(depth, physical_max_depth)

        # Configuration matches the paper's "Federated PC" setup
        # num_classes = num_clusters (The Latent Variable L)
        self.config = EinetConfig(
            num_features=num_features,
            num_channels=1,
            num_sums=num_sums,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
            depth=actual_depth,
            num_classes=num_clusters,  # Crucial: Each class represents a 'cluster' k
            leaf_type=Normal,
            layer_type="linsum",
            structure="top-down",
        )

        self.model = Einet(self.config).to(self.device)
        self.data_count = 0

    def train(self, X_local: np.ndarray, epochs=100, lr=None, batch_size=None):
        """
        Trains the local model with Mini-batches, LR Scheduler, and Early Stopping.
        """
        self.data_count = X_local.shape[0]
        self.model.train()
        use_lr = lr if lr is not None else self.lr
        use_bs = batch_size if batch_size is not None else self.batch_size

        # simple-einet expects [N, C, D] or [N, D]
        tensor_data = torch.tensor(X_local, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(
            dataset, batch_size=use_bs, shuffle=True, drop_last=False
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=use_lr, weight_decay=1e-3
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_loss = float("inf")
        patience_counter = 0
        patience_limit = 15

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for (batch_x,) in dataloader:
                optimizer.zero_grad()

                # Forward Pass
                ll_output = self.model(batch_x)

                # Handling Clusters/Classes
                if self.num_clusters > 1:
                    # LogSumExp over clusters to get P(X)
                    log_prob = torch.logsumexp(ll_output, dim=1)
                else:
                    log_prob = ll_output

                # NLL Loss
                loss = -log_prob.mean()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

                # Average loss for the epoch
            avg_loss = epoch_loss / max(1, batch_count)

            # 2. Scheduler Step
            scheduler.step(avg_loss)

            # 3. Early Stopping Check
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    # Optional: Print info if needed
                    # print(f"Client {self.client_id} converged at epoch {epoch}")
                    break

        self.model.eval()
        logging.info(
            f"Client {self.client_id} finished training. Final Loss: {best_loss:.4f}"
        )
        return self


class ServerSPN(FederatedSPNBase):
    """
    Server Node: The Federated Circuit (FC) Orchestrator.
    Implements the 'Arithmetic' to combine client distributions.
    """

    def __init__(
        self,
        global_num_features: int,
        scenario: str = "horizontal",
        num_clusters: int = 1,  # K (Latent clusters)
        device="cpu",
        threshold: float = None,  # use Adaptive Logic by default
    ):
        super().__init__(device)
        self.global_num_features = global_num_features
        self.scenario = scenario
        self.num_clusters = num_clusters
        # If threshold is provided, use it as a 'margin' for the permutation test
        self.threshold = threshold if threshold is not None else 0.005

        self.clients: List[ClientSPN] = []
        self.client_weights: List[float] = []  # For Horizontal mixing
        self.feature_map: Dict[
            int, List[int]
        ] = {}  # Maps client_id -> [feature_indices]

    def register_client(self, client: ClientSPN, feature_indices: List[int]):
        """
        Registers a trained client.
        feature_indices: The list of global feature indices this client owns.
        """
        self.clients.append(client)
        self.feature_map[id(client)] = feature_indices
        self.client_weights.append(client.data_count)

    def ci_test(
        self,
        x_idx: Union[int, List[int]],
        y_idx: Union[int, List[int]],
        z_idx: List[int],
        data_matrix: np.ndarray,
        num_permutations: int = 10,  # Number of shuffles to estimate noise
        sigma_threshold: float = 3.0,
    ) -> float:
        """
        The Oracle Function used by CDNOD.
        Computes CMI(X; Y | Z) using the Federated Circuit.

        Args:
            x_idx: Target variables
            y_idx: Target variables
            z_idx: Conditioning variables
            data_matrix: Proxy data (e.g. from the test set) to evaluate expectations.
        """
        # Input normalization
        if isinstance(x_idx, (int, np.integer)):
            x_idx = [x_idx]
        if isinstance(y_idx, (int, np.integer)):
            y_idx = [y_idx]
        if z_idx is None:
            z_idx = []
        z_idx = list(z_idx)

        # Convert data to tensor once
        if not isinstance(data_matrix, torch.Tensor):
            data_tensor = torch.tensor(data_matrix, dtype=torch.float32).to(self.device)
        else:
            data_tensor = data_matrix

        # 1. Calculate Observed CMI (Signal + Noise)
        cmi_obs = self._calculate_cmi_value(data_tensor, x_idx, y_idx, z_idx)

        # 2. Estimate Noise Distribution (Null Hypothesis)
        null_dist = []
        for _ in range(num_permutations):
            # Shuffle X to break dependency
            perm_indices = torch.randperm(data_tensor.size(0), device=self.device)
            data_perm = data_tensor.clone()
            for x_i in x_idx:
                data_perm[:, x_i] = data_tensor[perm_indices, x_i]

            # Compute null CMI
            val = self._calculate_cmi_value(data_perm, x_idx, y_idx, z_idx)
            null_dist.append(max(0.0, val))  # Clamp at 0

        # 3. Z-Score Test
        null_mean = np.mean(null_dist)
        null_std = np.std(null_dist) + 1e-6  # Avoid div/0

        z_score = (cmi_obs - null_mean) / null_std

        # Decision: Is the observation significantly distinct from noise?
        if z_score > sigma_threshold:
            return 0.0  # Dependent (Reject Null)
        else:
            return 1.0  # Independent (Accept Null)

    def _calculate_cmi_value(self, data_tensor, x_idx, y_idx, z_idx):
        """Helper to compute raw CMI value."""
        # CMI(X;Y|Z) ≈ E[ log P(X|Y,Z) - log P(X|Z) ]
        #            = E[ log P(X,Y,Z) - log P(Y,Z) - (log P(X,Z) - log P(Z)) ]
        #            = E[ log P(XYZ) - log P(XZ) - log P(YZ) + log P(Z) ]
        ll_xyz = self._federated_inference(data_tensor, x_idx + y_idx + z_idx)
        ll_xz = self._federated_inference(data_tensor, x_idx + z_idx)
        ll_yz = self._federated_inference(data_tensor, y_idx + z_idx)
        ll_z = self._federated_inference(data_tensor, z_idx) if z_idx else 0.0

        return ll_xyz - ll_xz - ll_yz + ll_z

    def _federated_inference(self, data: torch.Tensor, scope: List[int]) -> float:
        if self.scenario == "horizontal":
            return self._inference_horizontal(data, scope)
        elif self.scenario in ["vertical", "hybrid"]:
            return self._inference_vertical(data, scope)
        return 0.0

    def _get_client_log_prob(
        self, client: ClientSPN, global_data: torch.Tensor, global_scope: List[int]
    ):
        """
        Helper: Slices global data to match client's local features.
        """
        client_feats = self.feature_map[id(client)]
        batch_size = global_data.shape[0]

        # 1. Create a local tensor of the correct shape [N, D_local] with NaNs
        # This handles the "200 vs 4" mismatch error.
        local_data = torch.full(
            (batch_size, len(client_feats)), float("nan"), device=self.device
        )

        # 2. Identify intersection of request scope and client features
        # Map Global Index -> Local Index
        local_scope_indices = []
        for local_idx, global_idx in enumerate(client_feats):
            if global_idx in global_scope:
                # Fill observed data
                local_data[:, local_idx] = global_data[:, global_idx]
                local_scope_indices.append(local_idx)

        # 3. If client covers NONE of the requested variables, return 0.0 (Log Likelihood of 1)
        # However, for product aggregation, we might need valid shape.
        # simple-einet handles all-nan input as marginalizing everything -> 0.0 log prob.

        with torch.no_grad():
            return client.model(local_data)

    def _inference_horizontal(self, data: torch.Tensor, scope: List[int]) -> float:
        """
        Model: P(V) = Sum_k w_k P_k(V) (Mixture)
        """
        # 1. Weights
        total_count = sum(self.client_weights)
        log_weights = torch.tensor(
            [np.log(w / total_count) for w in self.client_weights], device=self.device
        )  # [K_clients]

        # 2. Query Clients
        # Each client returns [N, 1] log-probs
        client_lls = []
        for client in self.clients:
            # Horizontal: All clients have all features. Scope matches.
            ll = self._get_client_log_prob(client, data, scope)
            client_lls.append(ll)  # List of [N, 1]

        # Stack: [N, K_clients]
        stacked_lls = torch.cat(client_lls, dim=1)

        # 3. Mixture LogSumExp
        # log( Sum w_k exp(ll_k) )
        weighted_lls = stacked_lls + log_weights.unsqueeze(0)
        final_ll = torch.logsumexp(weighted_lls, dim=1)

        return final_ll.mean().item()

    def _inference_vertical(self, data: torch.Tensor, scope: List[int]) -> float:
        """
        Model: P(V) = Sum_c Prior(c) * Prod_k P_k(V_k | c) (Mixture of Products)
        Paper Assumption 2.
        """
        # Prior over clusters (Latent L). Assuming uniform for now (1/K)
        # or we could learn this.
        log_prior = -np.log(self.num_clusters)

        # We need to accumulate log-probs for each cluster c
        # Result shape goal: [N, Num_Clusters]

        # Initialize with Prior: [N, Num_Clusters]
        # We start with 0.0 (log(1)) because we are doing Product (Sum of Logs)
        batch_size = data.shape[0]
        cluster_accumulators = torch.zeros(
            (batch_size, self.num_clusters), device=self.device
        )

        for client in self.clients:
            # Query Client (Returns [N, K])
            client_cluster_lls = self._get_client_log_prob(client, data, scope)

            # Product Rule in Log Space = Sum
            cluster_accumulators += client_cluster_lls

            # Marginalize Latent Cluster
        final_vals = torch.logsumexp(cluster_accumulators + log_prior, dim=1)
        return final_vals.mean().item()


def auto_tune_spn_config(proxy_data, num_clusters=1, n_trials=15, device="cpu"):
    """
    Performs automated hyperparameter tuning on a slice of local data.
    """
    logging.info(
        f"[Auto-Tune] Starting optimization on {proxy_data.shape} proxy samples..."
    )

    # 1. Split Proxy Data into Train/Val (80/20)
    # Ensure we have enough data to split
    if len(proxy_data) < 10:
        logging.warning("[Auto-Tune] Not enough proxy data. Returning default config.")
        return {
            "num_sums": 10,
            "num_leaves": 20,
            "depth": 2,
            "lr": 0.05,
            "batch_size": 32,
        }

    train_data, val_data = train_test_split(proxy_data, test_size=0.2, random_state=42)
    val_tensor = torch.tensor(val_data, dtype=torch.float32).to(device)

    def objective(trial):
        # --- Suggest Hyperparameters ---
        num_sums = trial.suggest_int("num_sums", 5, 40)
        num_leaves = trial.suggest_int("num_leaves", 5, 40)
        depth = trial.suggest_int(
            "depth", 1, 4
        )  # 5 might be too deep for small features
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        # --- Instantiate ClientSPN ---
        # We use a dummy client_id=-1 since this is just for tuning
        client = ClientSPN(
            client_id=-1,
            num_features=proxy_data.shape[1],
            num_clusters=num_clusters,
            device=device,
            depth=depth,
            num_sums=num_sums,
            num_leaves=num_leaves,
            num_repetitions=5,
        )

        # --- Short Training Loop (Warm-up) ---
        # Train for only 5 epochs to get a performance signal quickly
        try:
            client.train(train_data, epochs=5, lr=lr, batch_size=batch_size)
        except Exception as e:
            # Prune invalid configs (e.g. diverging loss)
            return float("inf")

        # --- Evaluation ---
        client.model.eval()
        with torch.no_grad():
            ll_output = client.model(val_tensor)
            if num_clusters > 1:
                log_prob = torch.logsumexp(ll_output, dim=1)
            else:
                log_prob = ll_output

            # Minimize Negative Log-Likelihood
            val_nll = -log_prob.mean().item()

        return val_nll

    # Run Optuna Study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    logging.info(f"[Auto-Tune] Best Params: {study.best_params}")
    return study.best_params
