import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List
from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.normal import Normal


from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


class FederatedStructureLearner:
    """
    Learns a global variable dependency structure from federated metadata.
    Goal: Find an ordering of variables that places dependent variables close together,
    improving the effectiveness of the SPN's internal splitting rules.
    """

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.local_correlations = []

    def add_local_metadata(self, data: np.ndarray):
        """Clients call this to share their local correlation matrix."""
        # Use absolute correlation as proxy for dependence magnitude
        corr = np.abs(np.corrcoef(data, rowvar=False))
        # Handle NaNs (e.g. constant features)
        corr = np.nan_to_num(corr, nan=0.0)
        self.local_correlations.append(corr)

    def get_causal_order(self) -> np.ndarray:
        """
        Aggregates correlations and returns a dependency-aware variable ordering.
        Uses Hierarchical Clustering to group dependent variables.
        """
        if not self.local_correlations:
            return np.arange(self.num_features)

        # 1. Aggregate: Global Mean Dependency Matrix
        global_corr = np.mean(self.local_correlations, axis=0)

        # 2. Convert to Distance Matrix (1 - corr)
        # We want highly correlated variables to have low distance
        dist_matrix = 1.0 - global_corr
        np.fill_diagonal(dist_matrix, 0)

        # 3. Hierarchical Clustering (Ward's Method)
        # squareform converts the matrix to the compressed distance vector expected by linkage
        try:
            Z = linkage(squareform(dist_matrix, checks=False), method="ward")
            # 4. Extract Optimal Leaf Ordering
            order = leaves_list(Z)
            return order
        except Exception:
            # Fallback to identity order if clustering fails (e.g. too few features)
            return np.arange(self.num_features)


class LocalSPNWrapper(nn.Module):
    def __init__(
        self,
        num_features,
        device="cpu",
        depth=2,
        num_sums=20,
        num_leaves=20,
        num_repetitions=10,
        seed=None,
        variable_order=None,
    ):
        super().__init__()
        self.device = device
        self.mean = None
        self.std = None

        # Dependency-Aware Ordering
        if variable_order is not None:
            self.variable_order = torch.tensor(variable_order, dtype=torch.long).to(
                device
            )
            # Inverse order for sampling/denormalization
            self.inv_variable_order = torch.zeros_like(self.variable_order)
            self.inv_variable_order[self.variable_order] = torch.arange(
                num_features, device=device
            )
        else:
            self.variable_order = None

        # Ensure structural heterogeneity by seeding
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.config = EinetConfig(
            num_features=num_features,
            num_channels=1,
            num_sums=num_sums,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
            depth=depth,
            num_classes=1,
            leaf_type=Normal,
            layer_type="linsum",
            structure="top-down",
        )
        self.model = Einet(self.config).to(device)

    def _normalize(self, data):
        if self.mean is None or self.std is None:
            return data
        return (data - self.mean) / (self.std + 1e-6)

    def _permute(self, x):
        if self.variable_order is None:
            return x
        # x is [N, D]
        return x[:, self.variable_order]

    def _inv_permute(self, x):
        if self.variable_order is None:
            return x
        return x[:, self.inv_variable_order]

    def train_local(self, data, weights=None, epochs=50, lr=0.005):
        """
        Train this specific leaf on its data slice.
        weights: Optional [N] array of sample weights (for EM).
        """
        if len(data) < 5:
            return 0.0

        # Compute and store normalization stats
        if self.mean is None:
            self.mean = torch.tensor(data.mean(axis=0), dtype=torch.float32).to(
                self.device
            )
            self.std = torch.tensor(data.std(axis=0), dtype=torch.float32).to(
                self.device
            )

        data_t = torch.tensor(data, dtype=torch.float32).to(self.device)
        data_t = self._normalize(data_t)
        data_t = self._permute(data_t)  # Apply dependency-aware ordering

        if weights is not None:
            weights_t = torch.tensor(weights, dtype=torch.float32).to(self.device)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        final_ll = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            ll = self.model(data_t)

            if weights is not None:
                loss = -(ll * weights_t.unsqueeze(1)).sum() / (weights_t.sum() + 1e-9)
            else:
                loss = -ll.mean()

            loss.backward()
            optimizer.step()
            final_ll = -loss.item()
        return final_ll

    def sample(self, n):
        samples = self.model.sample(n)
        samples = self._inv_permute(samples)  # Restore original variable order
        if self.mean is not None and self.std is not None:
            samples = samples * (self.std + 1e-6) + self.mean
        return samples

    def log_prob(self, x):
        # 1. Normalize and Permute
        x_norm = self._normalize(x)
        x_perm = self._permute(x_norm)

        # 2. Forward pass through Einet
        ll = self.model(x_perm)

        # 3. Log-Jacobian Correction
        if self.std is not None:
            # Jacobian is based on scale sigma.
            # Mask is based on ORIGINAL x (not permuted)
            mask = (~torch.isnan(x)).float()
            log_sigma = torch.log(self.std + 1e-6)
            log_det_jacobian = -(log_sigma * mask).sum(dim=1, keepdim=True)
            ll = ll + log_det_jacobian
        return ll


class FederatedProduct(nn.Module):
    """
    Vertical Federated SPN Component.
    Models P(X) = Prod_k P_k(X_k).
    """

    def __init__(
        self,
        clients: List[LocalSPNWrapper],
        feature_map: Dict[int, List[int]],
        device="cpu",
    ):
        super().__init__()
        self.clients = nn.ModuleList(clients)
        self.feature_map = feature_map
        self.device = device

    def log_prob(self, x):
        client_lls = []
        for i, client in enumerate(self.clients):
            indices = self.feature_map[i]
            x_local = x[:, indices]
            client_lls.append(client.log_prob(x_local))

        # Sum log-probs (Product in prob space)
        ll_stack = torch.cat(client_lls, dim=1)
        return torch.sum(ll_stack, dim=1, keepdim=True)


class GlobalFedSPN(nn.Module):
    """
    Global Federated SPN (Mixture).
    Models P(X) = Sum_c w_c * Component_c(X).
    Components can be LocalSPNWrapper (Horizontal) or FederatedProduct (Vertical Latent).
    """

    def __init__(
        self,
        components: List[nn.Module],
        weights: List[float] = None,
        feature_map: Dict[int, List[int]] = None,
        strategy: str = "mixture",
        device="cpu",
    ):
        super().__init__()
        self.components = nn.ModuleList(components)
        self.device = device
        self.strategy = strategy.lower()
        self.feature_map = feature_map

        if weights is None:
            n = len(components)
            weights = [1.0 / n] * n

        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)

        # Backward compatibility aliases
        self.clients = self.components

    def train_weights_em(self, x, epochs=10, lr=0.01):
        """
        Refine the mixture weights using EM (Expectation-Maximization) on the server.
        Optimizes P(X) = Sum_k w_k * Component_k(X) w.r.t w_k.
        Useful for Vertical FL to better align the latent components.
        """
        if self.strategy != "mixture":
            logging.warning("EM weight training only valid for 'mixture' strategy.")
            return

        prev_ll = -float("inf")

        with torch.no_grad():
            for epoch in range(epochs):
                # 1. E-Step: Compute Responsibilities (Posteriors)
                comp_lls = []
                for i, c in enumerate(self.components):
                    if self.feature_map is not None:
                        idx = self.feature_map[i]
                        x_c = x[:, idx]
                    else:
                        x_c = x
                    comp_lls.append(c.log_prob(x_c))

                ll_stack = torch.cat(comp_lls, dim=1)

                # Log Prior
                log_w = torch.log(self.weights + 1e-9).unsqueeze(0)

                # Log Joint: [N, K]
                log_joint = ll_stack + log_w

                # Log Marginal (Evidence): [N, 1]
                log_evidence = torch.logsumexp(log_joint, dim=1, keepdim=True)

                # Log Posterior: [N, K]
                log_posterior = log_joint - log_evidence

                # Monitor convergence
                current_ll = log_evidence.mean().item()
                if abs(current_ll - prev_ll) < 1e-4:
                    break
                prev_ll = current_ll

                # 2. M-Step: Update Weights
                responsibilities = torch.exp(log_posterior)
                new_weights = responsibilities.mean(dim=0)

                # Update
                self.weights = new_weights.detach()

        logging.info(f"[FedPC] EM Refined Weights: {self.weights.cpu().numpy()}")

    def log_prob(self, x):
        """
        Computes log P(X).
        """
        if self.strategy == "product":
            # Direct Product over components
            comp_lls = []
            for i, c in enumerate(self.components):
                if self.feature_map is not None:
                    idx = self.feature_map[i]
                    x_c = x[:, idx]
                else:
                    x_c = x
                comp_lls.append(c.log_prob(x_c))
            return torch.sum(torch.cat(comp_lls, dim=1), dim=1, keepdim=True)

        # Default: Mixture (Sum Node)
        comp_lls = []
        for i, c in enumerate(self.components):
            if self.feature_map is not None:
                idx = self.feature_map[i]
                x_c = x[:, idx]
            else:
                x_c = x
            comp_lls.append(c.log_prob(x_c))

        ll_stack = torch.cat(comp_lls, dim=1)
        log_w = torch.log(self.weights + 1e-9).unsqueeze(0)
        return torch.logsumexp(ll_stack + log_w, dim=1, keepdim=True)

    def log_prob_conditional_u(self, x, u_idx):
        """
        Routing for Horizontal scenario where components ARE clients.
        """
        if 0 <= u_idx < len(self.components):
            if self.feature_map is not None:
                idx = self.feature_map[u_idx]
                x_c = x[:, idx]
            else:
                x_c = x
            return self.components[u_idx].log_prob(x_c)
        else:
            raise ValueError(f"Index {u_idx} out of bounds")


def auto_tune_spn_config(proxy_data, num_clusters=1, n_trials=15, device="cpu"):
    # ... placeholder or original implementation if needed ...
    # For now, returning default dict as I overwrote the file
    # But wait, I should keep the original implementation if possible.
    # The read_file showed it at the end. I will copy it back.
    import optuna
    from sklearn.model_selection import train_test_split

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

        # IMPORTANT: Use LocalSPNWrapper here!
        client = LocalSPNWrapper(
            num_features=proxy_data.shape[1],
            device=device,
            depth=depth,
            num_sums=num_sums,
            num_leaves=num_leaves,
            num_repetitions=5,
        )

        try:
            client.train_local(train_data, epochs=5, lr=lr)
        except Exception:
            return float("inf")

        client.model.eval()
        with torch.no_grad():
            ll_output = client.log_prob(val_tensor)
            val_nll = -ll_output.mean().item()

        return val_nll

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logging.info(f"[Auto-Tune] Best Params: {study.best_params}")
    return study.best_params
