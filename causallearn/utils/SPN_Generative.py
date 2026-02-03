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


# --- Base Classes ---
class FederatedSPNBase:
    def __init__(self, device="cpu"):
        self.device = device


class ClientSPN(FederatedSPNBase):
    def __init__(
        self,
        client_id,
        num_features,
        num_clusters=1,
        device="cpu",
        depth=2,
        num_sums=10,
        num_leaves=10,
        num_repetitions=5,
        lr=0.01,
        batch_size=32,
    ):
        super().__init__(device)
        self.client_id = client_id
        self.num_clusters = num_clusters
        self.lr = lr
        self.batch_size = batch_size

        # Calculate safe depth based on feature count
        physical_max_depth = int(np.floor(np.log2(num_features)))
        actual_depth = min(depth, physical_max_depth) if depth else physical_max_depth
        if actual_depth < 1:
            actual_depth = 1

        self.config = EinetConfig(
            num_features=num_features,
            num_channels=1,
            num_sums=num_sums,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
            depth=actual_depth,
            num_classes=num_clusters,
            leaf_type=Normal,
            layer_type="linsum",
            structure="top-down",
        )
        self.model = Einet(self.config).to(self.device)
        self.data_count = 0

    def train_epoch(self, X_local, weights=None, lr=None):
        self.model.train()
        use_lr = lr if lr else self.lr
        optimizer = torch.optim.Adam(self.model.parameters(), lr=use_lr)

        data = torch.tensor(X_local, dtype=torch.float32).to(self.device)
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        optimizer.zero_grad()
        ll_output = self.model(data)

        if self.num_clusters > 1 and weights is not None:
            # Weighted MLE (EM M-Step for Vertical/Hybrid)
            # Weights are responsibilities from the E-Step
            loss = -(weights * ll_output).sum(dim=1).mean()
        else:
            # Standard MLE
            log_prob = (
                torch.logsumexp(ll_output, dim=1)
                if self.num_clusters > 1
                else ll_output
            )
            loss = -log_prob.mean()

        loss.backward()
        optimizer.step()
        return loss.item()

    def get_weights(self):
        return copy.deepcopy(self.model.state_dict())

    def set_weights(self, weights):
        self.model.load_state_dict(weights)


class ServerSPN(FederatedSPNBase):
    def __init__(
        self, global_num_features, scenario="horizontal", num_clusters=1, device="cpu"
    ):
        super().__init__(device)
        self.global_num_features = global_num_features
        self.scenario = scenario
        self.num_clusters = num_clusters
        self.clients = []
        self.feature_map = {}

    def register_client(self, client: ClientSPN, feature_indices: List[int]):
        self.clients.append(client)
        self.feature_map[id(client)] = feature_indices

    def perform_e_step(self, data_matrix, scope_ranges):
        """
        Standard EM E-Step for Vertical/Hybrid scenarios.
        Calculates posterior probabilities (responsibilities) of latent clusters.
        """
        log_prior = -np.log(self.num_clusters)
        data_t = torch.tensor(data_matrix, dtype=torch.float32).to(self.device)
        batch_size = data_t.shape[0]
        cluster_accumulators = torch.zeros(
            (batch_size, self.num_clusters), device=self.device
        )

        with torch.no_grad():
            for i, client in enumerate(self.clients):
                client_scope = scope_ranges[i]
                # Manual client forward pass on their specific columns
                local_data = data_t[:, client_scope]
                client_ll = client.model(local_data)
                cluster_accumulators += client_ll

        log_joint = cluster_accumulators + log_prior
        log_marginal = torch.logsumexp(log_joint, dim=1, keepdim=True)
        log_posterior = log_joint - log_marginal
        return torch.exp(log_posterior).cpu().numpy()

    def generate_global_synthetic_data(self, n_samples=5_000):
        """
        The Core of Generative Discovery.
        Returns:
            X (np.float32): Synthetic data features.
            C (np.int64): Context/Client indices (Crucial for CDNOD).
        """
        logging.info(f"Generating {n_samples} samples (Scenario: {self.scenario})...")

        if self.scenario == "horizontal":
            # Horizontal: Clients model different rows.
            # We sample proportionally from each client to form the global proxy.
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
                    # Create Integer Context Vector
                    all_C.append(np.full((samples_per_client, 1), k, dtype=int))

            # Handle remainder samples
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
            # Vertical/Hybrid: Clients model different features (columns).
            # We assume independence between column blocks for generation (Simplified)
            # OR we assume the trained EM latent structure holds.
            col_parts = []

            # Sort clients by their first feature index to stack columns in correct order 0..D
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
            # For Vertical, 'Context' is usually global (Stationary) or just 0
            C_syn = np.zeros((n_samples, 1), dtype=int)

            X_syn += np.random.normal(0, 1e-6, X_syn.shape)

            return X_syn, C_syn

        # Fallback
        return np.zeros((n_samples, self.global_num_features)), np.zeros(
            (n_samples, 1), dtype=int
        )


def auto_tune_spn_config(proxy_data, num_clusters=1, n_trials=5, device="cpu"):
    # Simplified Auto-Tuner using Optuna
    proxy_data = np.nan_to_num(proxy_data, nan=0.0)

    # Don't auto-tune if data is tiny
    if len(proxy_data) < 50:
        return {
            "num_sums": 10,
            "num_leaves": 10,
            "depth": 1,
            "lr": 0.05,
            "batch_size": 32,
        }

    def objective(trial):
        num_sums = trial.suggest_int("num_sums", 10, 30)
        num_leaves = trial.suggest_int("num_leaves", 10, 20)
        depth = trial.suggest_int("depth", 1, 3)
        lr = trial.suggest_float("lr", 1e-3, 0.05, log=True)

        c = ClientSPN(
            -1,
            proxy_data.shape[1],
            num_clusters,
            device,
            depth,
            num_sums,
            num_leaves,
            5,
            lr,
            32,
        )
        try:
            # Quick 1-epoch test
            loss = c.train_epoch(proxy_data)
            return loss
        except:
            return float("inf")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
