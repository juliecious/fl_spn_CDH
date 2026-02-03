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
        depth: int = None,
        num_sums: int = 10,
        num_leaves: int = 20,
        num_repetitions: int = 5,
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
        physical_max_depth = int(np.floor(np.log2(num_features)))
        actual_depth = min(depth, physical_max_depth) if depth else physical_max_depth

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
        Standard Local Training (for initialization or baselines).
        """
        self.data_count = X_local.shape[0]
        self.model.train()
        use_lr = lr if lr is not None else self.lr
        use_bs = batch_size if batch_size is not None else self.batch_size

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
                ll_output = self.model(batch_x)

                if self.num_clusters > 1:
                    log_prob = torch.logsumexp(ll_output, dim=1)
                else:
                    log_prob = ll_output

                loss = -log_prob.mean()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / max(1, batch_count)
            scheduler.step(avg_loss)

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    break

        self.model.eval()
        logging.info(
            f"Client {self.client_id} finished training. Final Loss: {best_loss:.4f}"
        )
        return self

    def get_weights(self):
        """Return state dict for FedAvg."""
        return copy.deepcopy(self.model.state_dict())

    def set_weights(self, weights):
        """Load global weights."""
        self.model.load_state_dict(weights)

    def train_epoch(self, X_local, weights=None, lr=0.05):
        """
        Runs ONE epoch of training.
        Args:
            X_local: Local data [N, D]
            weights: (Optional) Responsibilities [N, K] for Vertical EM.
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Convert to tensor
        data = torch.tensor(X_local, dtype=torch.float32).to(self.device)

        # If performing EM (Vertical), we need the weights (responsibilities)
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        optimizer.zero_grad()

        # Forward pass: [N, num_clusters]
        ll_output = self.model(data)

        if self.num_clusters > 1 and weights is not None:
            # --- M-Step (Vertical/Hybrid) ---
            # Maximize: Sum_k gamma_ik * log P(X_i | Z=k)
            # Loss = - Mean( Sum( weights * ll_output ) )
            loss = -(weights * ll_output).sum(dim=1).mean()
        else:
            # --- Standard MLE (Horizontal/Local) ---
            # Marginalize out Z: log Sum_k exp(ll_k)
            log_prob = torch.logsumexp(ll_output, dim=1)
            loss = -log_prob.mean()

        loss.backward()
        optimizer.step()
        return loss.item()


class ServerSPN(FederatedSPNBase):
    """
    Server Node: The Federated Circuit (FC) Orchestrator.
    """

    def __init__(
        self,
        global_num_features: int,
        scenario: str = "horizontal",
        num_clusters: int = 1,
        device="cpu",
        threshold: float = None,
    ):
        super().__init__(device)
        self.global_num_features = global_num_features
        self.scenario = scenario
        self.num_clusters = num_clusters
        self.threshold = threshold if threshold is not None else 0.005
        self.clients: List[ClientSPN] = []
        self.client_weights: List[float] = []
        self.feature_map: Dict[int, List[int]] = {}

    def register_client(self, client: ClientSPN, feature_indices: List[int]):
        self.clients.append(client)
        self.feature_map[id(client)] = feature_indices
        self.client_weights.append(client.data_count)

    def ci_test(
        self,
        x_idx: Union[int, List[int]],
        y_idx: Union[int, List[int]],
        z_idx: List[int],
        data_matrix: np.ndarray,
        num_permutations: int = 10,
        sigma_threshold: float = 3.0,
    ) -> float:
        if isinstance(x_idx, (int, np.integer)):
            x_idx = [x_idx]
        if isinstance(y_idx, (int, np.integer)):
            y_idx = [y_idx]
        if z_idx is None:
            z_idx = []
        z_idx = list(z_idx)

        if not isinstance(data_matrix, torch.Tensor):
            data_tensor = torch.tensor(data_matrix, dtype=torch.float32).to(self.device)
        else:
            data_tensor = data_matrix

        # 1. Pre-calculate terms that don't change (Static)
        # Optimization: These are calculated ONLY ONCE.
        ll_z = self._federated_inference(data_tensor, z_idx) if z_idx else 0.0
        ll_yz = self._federated_inference(data_tensor, y_idx + z_idx)

        # 2. Calculate Observed CMI (Using the static terms)
        ll_xyz = self._federated_inference(data_tensor, x_idx + y_idx + z_idx)
        ll_xz = self._federated_inference(data_tensor, x_idx + z_idx)

        # REFACTOR: Construct cmi_obs manually to avoid re-computing ll_z/ll_yz
        cmi_obs = ll_xyz - ll_xz - ll_yz + ll_z

        # 3. Permutation Loop (Using the static terms)
        null_dist = []
        for _ in range(num_permutations):
            perm_indices = torch.randperm(data_tensor.size(0), device=self.device)
            data_perm = data_tensor.clone()
            for x_i in x_idx:
                data_perm[:, x_i] = data_tensor[perm_indices, x_i]

            # Only calculate the terms that changed
            ll_xyz_perm = self._federated_inference(data_perm, x_idx + y_idx + z_idx)
            ll_xz_perm = self._federated_inference(data_perm, x_idx + z_idx)

            val = ll_xyz_perm - ll_xz_perm - ll_yz + ll_z
            null_dist.append(max(0.0, val))

        null_mean = np.mean(null_dist)
        null_std = np.std(null_dist) + 1e-6
        z_score = (cmi_obs - null_mean) / null_std

        if z_score > sigma_threshold:
            return 0.0  # Dependent
        else:
            return 1.0  # Independent

    def _calculate_cmi_value(self, data_tensor, x_idx, y_idx, z_idx):
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
        client_feats = self.feature_map[id(client)]
        batch_size = global_data.shape[0]
        local_data = torch.full(
            (batch_size, len(client_feats)), float("nan"), device=self.device
        )

        for local_idx, global_idx in enumerate(client_feats):
            if global_idx in global_scope:
                local_data[:, local_idx] = global_data[:, global_idx]

        with torch.no_grad():
            return client.model(local_data)

    def _inference_horizontal(self, data: torch.Tensor, scope: List[int]) -> float:
        """
        Federated Ensemble (Voting)
        Instead of averaging weights (which destroys structure), we average predictions.
        """
        # 1. Query every client for their local log-probability
        client_lls = []
        for client in self.clients:
            local_ll = self._get_client_log_prob(client, data, scope)
            # Ensure shape is [N]
            if client.num_clusters > 1:
                local_ll = torch.logsumexp(local_ll, dim=1)
            client_lls.append(local_ll)

        # 2. Stack and Average in Log-Space (LogSumExp)
        # Formula: log( 1/K * sum(exp(ll_k)) ) = logsumexp(ll_k) - log(K)
        stacked_lls = torch.stack(client_lls, dim=0)  # [K, N]
        log_k = np.log(len(self.clients))

        ensemble_ll = torch.logsumexp(stacked_lls, dim=0) - log_k

        return ensemble_ll.mean().item()

    def _inference_vertical(self, data: torch.Tensor, scope: List[int]) -> float:
        """
        Model: P(V) = Sum_c Prior(c) * Prod_k P_k(V_k | c)
        """
        log_prior = -np.log(self.num_clusters)
        batch_size = data.shape[0]
        cluster_accumulators = torch.zeros(
            (batch_size, self.num_clusters), device=self.device
        )

        for client in self.clients:
            # Query Client (Returns [N, K])
            client_cluster_lls = self._get_client_log_prob(client, data, scope)
            cluster_accumulators += client_cluster_lls

        final_vals = torch.logsumexp(cluster_accumulators + log_prior, dim=1)
        return final_vals.mean().item()

    def perform_fedavg(self):
        """Aggregates client weights (Horizontal FL)."""
        global_dict = self.clients[0].get_weights()
        total_samples = sum(self.client_weights)

        # Weighted Average
        for k in global_dict.keys():
            global_dict[k] = global_dict[k] * (self.client_weights[0] / total_samples)

        for i in range(1, len(self.clients)):
            w_local = self.clients[i].get_weights()
            weight_factor = self.client_weights[i] / total_samples
            for k in global_dict.keys():
                global_dict[k] += w_local[k] * weight_factor

        # Broadcast back
        for client in self.clients:
            client.set_weights(global_dict)

    def perform_e_step(self, data_matrix, scope_ranges):
        """Calculates Global Posteriors (Responsibilities) for Vertical EM."""
        log_prior = -np.log(self.num_clusters)
        data_t = torch.tensor(data_matrix, dtype=torch.float32).to(self.device)
        batch_size = data_t.shape[0]
        cluster_accumulators = torch.zeros(
            (batch_size, self.num_clusters), device=self.device
        )

        with torch.no_grad():
            for i, client in enumerate(self.clients):
                client_scope = scope_ranges[i]
                client_ll = self._get_client_log_prob(client, data_t, client_scope)
                cluster_accumulators += client_ll

        log_joint = cluster_accumulators + log_prior
        log_marginal = torch.logsumexp(log_joint, dim=1, keepdim=True)
        log_posterior = log_joint - log_marginal
        responsibilities = torch.exp(log_posterior).cpu().numpy()
        return responsibilities

    def generate_global_synthetic_data(self, n_samples=2000):
        """
        Robust Data Generation. Returns X (float) and C (int).
        """
        logging.info(f"Generating {n_samples} samples (Scenario: {self.scenario})...")

        if self.scenario == "horizontal":
            # Round-Robin Sampling from Clients
            samples_per_client = n_samples // len(self.clients)
            all_X, all_C = [], []

            for k, client in enumerate(self.clients):
                with torch.no_grad():
                    s = (
                        client.model.sample(num_samples=samples_per_client)
                        .cpu()
                        .numpy()
                    )
                    if s.ndim == 3:
                        s = s.squeeze(1)
                    all_X.append(s)
                    all_C.append(np.full((samples_per_client, 1), k, dtype=int))

            # Remainder
            rem = n_samples - sum(len(x) for x in all_X)
            if rem > 0:
                with torch.no_grad():
                    s = self.clients[0].model.sample(num_samples=rem).cpu().numpy()
                    if s.ndim == 3:
                        s = s.squeeze(1)
                    all_X.append(s)
                    all_C.append(np.full((rem, 1), 0, dtype=int))

            return np.concatenate(all_X, axis=0), np.concatenate(all_C, axis=0)

        elif self.scenario in ["vertical", "hybrid"]:
            # Feature Stacking
            col_parts = []
            # Sort clients by their first feature index to stack correctly
            sorted_clients = sorted(
                self.clients, key=lambda c: self.feature_map[id(c)][0]
            )

            for client in sorted_clients:
                with torch.no_grad():
                    s = client.model.sample(num_samples=n_samples).cpu().numpy()
                    if s.ndim == 3:
                        s = s.squeeze(1)
                    col_parts.append(s)

            X_syn = np.hstack(col_parts)
            # Default Context = 0 for Vertical (or random if you want to simulate heterogeneity)
            C_syn = np.zeros((n_samples, 1), dtype=int)
            return X_syn, C_syn

        return np.zeros((n_samples, self.global_num_features)), np.zeros(
            (n_samples, 1), dtype=int
        )


def auto_tune_spn_config(proxy_data, num_clusters=1, n_trials=15, device="cpu"):
    logging.info(
        f"[Auto-Tune] Starting optimization on {proxy_data.shape} proxy samples..."
    )

    # SAFETY: Handle NaNs in proxy data
    proxy_data = np.nan_to_num(proxy_data, nan=0.0)

    if len(proxy_data) < 10:
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
        num_sums = trial.suggest_int("num_sums", 20, 30)
        num_leaves = trial.suggest_int("num_leaves", 10, 25)
        depth = trial.suggest_int("depth", 2, 4)
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

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

        try:
            client.train(train_data, epochs=5, lr=lr, batch_size=batch_size)
        except Exception:
            return float("inf")

        client.model.eval()
        with torch.no_grad():
            ll_output = client.model(val_tensor)
            if num_clusters > 1:
                log_prob = torch.logsumexp(ll_output, dim=1)
            else:
                log_prob = ll_output
            val_nll = -log_prob.mean().item()

        return val_nll

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logging.info(f"[Auto-Tune] Best Params: {study.best_params}")
    return study.best_params
