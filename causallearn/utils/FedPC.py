import io
import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.normal import Normal


from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


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
        # Use Spearman Rank Correlation to capture non-linear (monotonic) dependencies
        # This is more robust for SPN structure learning than Pearson correlation
        corr, _ = spearmanr(data, axis=0)

        # Take absolute value as we only care about dependency strength
        corr = np.abs(corr)

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
        self.seed = seed

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
        # Ensure 2D for consistent indexing
        if x.ndim == 1:
            return x[self.variable_order]
        return x[:, self.variable_order]

    def _inv_permute(self, x):
        if self.variable_order is None:
            return x
        if x.ndim == 1:
            return x[self.inv_variable_order]
        return x[:, self.inv_variable_order]

    def train_local(self, data, weights=None, epochs=50, lr=0.005, l1_weight=1e-4):
        """
        Train this specific leaf on its data slice with L1 Sparsity Penalty.
        weights: Optional [N] array of sample weights (for EM).
        l1_weight: Weight for the L1 sparsity penalty on sum weights.
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

        # Extract cluster and client indices from seed (seed = h * 10 + k)
        cluster_id = self.seed // 10 if self.seed is not None else -1
        client_id = self.seed % 10 if self.seed is not None else -1

        final_ll = 0.0
        loss_history = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            ll = self.model(data_t)

            if weights is not None:
                nll_loss = -(ll * weights_t.unsqueeze(1)).sum() / (
                    weights_t.sum() + 1e-9
                )
            else:
                nll_loss = -ll.mean()

            # L1 Sparsity Penalty on Sum Weights
            l1_penalty = 0.0
            for name, param in self.model.named_parameters():
                if "sum" in name and "weight" in name:
                    l1_penalty += torch.norm(param, p=1)

            loss = nll_loss + l1_weight * l1_penalty

            loss.backward()
            optimizer.step()
            final_ll = -nll_loss.item()
            loss_history.append(loss.item())

            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                logging.debug(
                    f"Cluster {cluster_id}, Client {client_id}, Epoch {epoch + 1}/{epochs}: "
                    f"Loss={loss.item():.4f}"
                )

        # Log final training loss
        logging.info(
            f"Cluster {cluster_id}, Client {client_id}: Final Loss={loss_history[-1]:.4f} "
            f"after {epochs} epochs"
        )

        # Check if loss decreased in last 20 epochs
        if len(loss_history) >= 20:
            last_20_losses = loss_history[-20:]
            if (
                min(last_20_losses) >= loss_history[-20]
            ):  # No improvement in last 20 epochs
                logging.warning(
                    f"Cluster {cluster_id}, Client {client_id}: Loss did not decrease "
                    f"in last 20 epochs (started at {loss_history[-20]:.4f}, ended at {loss_history[-1]:.4f})"
                )

        return final_ll

    def sample(self, n):
        samples = self.model.sample(n)
        # Force 2D [n, num_features]
        samples = samples.view(n, -1)
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


class UnivariateSPNWrapper(nn.Module):
    """
    Handles 1-dimensional features using a simple Gaussian Mixture Model (GMM).
    Standard PyTorch implementation to avoid simple-einet overhead for 1D.
    """

    def __init__(self, device="cpu", num_sums=20, num_leaves=20, seed=None):
        super().__init__()
        self.device = device
        self.K = num_leaves  # Number of components
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # GMM Parameters
        self.logits = nn.Parameter(torch.randn(self.K))
        self.means = nn.Parameter(torch.randn(self.K))
        self.log_stds = nn.Parameter(torch.randn(self.K))

        # Normalization stats
        self.data_mean = None
        self.data_std = None

        self.to(device)

    def _normalize(self, data):
        if self.data_mean is None or self.data_std is None:
            return data
        return (data - self.data_mean) / (self.data_std + 1e-6)

    def train_local(self, data, weights=None, epochs=50, lr=0.01):
        if len(data) < 5:
            return 0.0

        # Compute normalization stats
        if self.data_mean is None:
            self.data_mean = torch.tensor(data.mean(), dtype=torch.float32).to(
                self.device
            )
            self.data_std = torch.tensor(data.std(), dtype=torch.float32).to(
                self.device
            )

        data_t = torch.tensor(data, dtype=torch.float32).to(self.device).view(-1, 1)
        data_t = self._normalize(data_t)  # [N, 1]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()

        # Extract cluster and client indices from seed (seed = h * 10 + k)
        cluster_id = self.seed // 10 if self.seed is not None else -1
        client_id = self.seed % 10 if self.seed is not None else -1

        final_ll = 0.0
        loss_history = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            # log P(x|k)
            # -0.5 * log(2pi) - log_std - 0.5 * ((x-mu)/std)^2
            std = torch.exp(self.log_stds)
            var = std.pow(2)
            log_scale = np.log(np.sqrt(2 * np.pi))

            # [N, K]
            diff = data_t - self.means.unsqueeze(0)
            log_p_xk = (
                -log_scale
                - self.log_stds.unsqueeze(0)
                - 0.5 * (diff.pow(2) / var.unsqueeze(0))
            )

            # log P(k)
            log_pi = torch.log_softmax(self.logits, dim=0)

            # log P(x) = logsumexp(log P(k) + log P(x|k))
            log_prob = torch.logsumexp(log_pi.unsqueeze(0) + log_p_xk, dim=1)

            if weights is not None:
                weights_t = torch.tensor(weights, dtype=torch.float32).to(self.device)
                loss = -(log_prob * weights_t).sum() / (weights_t.sum() + 1e-9)
            else:
                loss = -log_prob.mean()

            loss.backward()
            optimizer.step()
            final_ll = -loss.item()
            loss_history.append(loss.item())

            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                logging.debug(
                    f"Cluster {cluster_id}, Client {client_id}, Epoch {epoch + 1}/{epochs}: "
                    f"Loss={loss.item():.4f} (Univariate GMM)"
                )

        # Log final training loss
        logging.info(
            f"Cluster {cluster_id}, Client {client_id}: Final Loss={loss_history[-1]:.4f} "
            f"after {epochs} epochs (Univariate GMM)"
        )

        # Check if loss decreased in last 20 epochs
        if len(loss_history) >= 20:
            last_20_losses = loss_history[-20:]
            if (
                min(last_20_losses) >= loss_history[-20]
            ):  # No improvement in last 20 epochs
                logging.warning(
                    f"Cluster {cluster_id}, Client {client_id}: Loss did not decrease "
                    f"in last 20 epochs (started at {loss_history[-20]:.4f}, ended at {loss_history[-1]:.4f}) "
                    f"(Univariate GMM)"
                )

        return final_ll

    def log_prob(self, x):
        self.eval()
        with torch.no_grad():
            x_t = x.view(-1, 1)
            x_norm = self._normalize(x_t)

            std = torch.exp(self.log_stds)
            var = std.pow(2)
            log_scale = np.log(np.sqrt(2 * np.pi))

            diff = x_norm - self.means.unsqueeze(0)
            log_p_xk = (
                -log_scale
                - self.log_stds.unsqueeze(0)
                - 0.5 * (diff.pow(2) / var.unsqueeze(0))
            )

            log_pi = torch.log_softmax(self.logits, dim=0)
            ll = torch.logsumexp(log_pi.unsqueeze(0) + log_p_xk, dim=1, keepdim=True)

            # Jacobian correction
            if self.data_std is not None:
                log_sigma = torch.log(self.data_std + 1e-6)
                ll = ll - log_sigma
            return ll

    def sample(self, n):
        with torch.no_grad():
            # 1. Sample components
            probs = torch.softmax(self.logits, dim=0)
            comp_indices = torch.multinomial(probs, n, replacement=True)

            # 2. Sample from Gaussians
            means = self.means[comp_indices]
            stds = torch.exp(self.log_stds)[comp_indices]

            samples = torch.normal(means, stds).view(n, 1)

            # 3. Denormalize
            if self.data_mean is not None:
                samples = samples * (self.data_std + 1e-6) + self.data_mean

            return samples

    def get_size_bytes(self) -> int:
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return buffer.tell()


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

    def sample(self, n: int) -> torch.Tensor:
        """
        Sample from the product distribution.
        Since P(X) = Prod P_k(X_k), we sample each component and concatenate.
        """
        # Determine total features
        all_indices = []
        for v in self.feature_map.values():
            all_indices.extend(v)
        num_features = len(set(all_indices))

        samples = torch.zeros(n, num_features, device=self.device)
        for i, client in enumerate(self.clients):
            indices = self.feature_map[i]
            client_samples = client.sample(n)
            # Ensure client_samples is 2D
            client_samples = client_samples.view(n, -1)
            samples[:, indices] = client_samples
        return samples.view(n, -1)

    def get_size_bytes(self) -> int:
        """Sum of client sizes."""
        return sum(c.get_size_bytes() for c in self.clients)


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

    def sample(self, n: int) -> torch.Tensor:
        """
        Sample from the mixture distribution.
        1. Choose component based on weights.
        2. Sample from component.
        """
        if n <= 0:
            return torch.tensor([], device=self.device)

        # 1. Choose components
        comp_indices = torch.multinomial(self.weights, n, replacement=True)
        unique_comps, counts = torch.unique(comp_indices, return_counts=True)

        # Determine total features
        # If Horizontal, components have same features. If Vertical, they might differ.
        # Product components (Vertical) already handle internal feature maps.
        # For simplicity, we sample and fill.

        # Determine dimensionality
        if self.feature_map is not None:
            all_indices = []
            for v in self.feature_map.values():
                all_indices.extend(v)
            num_features = len(set(all_indices))
        else:
            # Assume all components have same features (Horizontal)
            # Sample from first to get shape
            test_sample = self.components[0].sample(1)
            num_features = test_sample.shape[1]

        samples = torch.zeros(n, num_features, device=self.device)

        # Track filling
        start_idx = 0
        for comp_idx, count in zip(unique_comps, counts):
            c = self.components[comp_idx.item()]
            comp_samples = c.sample(count.item())
            # Ensure 2D
            comp_samples = comp_samples.view(count.item(), -1)

            end_idx = start_idx + count.item()

            if self.feature_map is not None:
                indices = self.feature_map[comp_idx.item()]
                samples[start_idx:end_idx, indices] = comp_samples
            else:
                samples[start_idx:end_idx, :] = comp_samples

            start_idx = end_idx

        return samples.view(n, -1)

    def log_prob_conditional_u(self, x, u_idx):
        """
        Routing for Horizontal scenario where components ARE clients.

        Args:
            x: Feature data without context column, shape (batch_size, d)
            u_idx: Client index to condition on

        Returns:
            Log probability from the specified client's local SPN

        Note: Local SPNs were trained with context column appended.
        We reconstruct the augmented data [x, u_idx] before evaluation.
        """
        if 0 <= u_idx < len(self.components):
            # Local SPNs expect augmented data [features, context]
            # Reconstruct by appending the context value
            u_col = torch.full((x.shape[0], 1), u_idx, dtype=x.dtype, device=x.device)
            x_aug = torch.cat([x, u_col], dim=1)

            if self.feature_map is not None:
                idx = self.feature_map[u_idx]
                x_c = x_aug[:, idx]
            else:
                x_c = x_aug
            return self.components[u_idx].log_prob(x_c)
        else:
            raise ValueError(f"Index {u_idx} out of bounds")

    def get_total_communication_cost(self) -> int:
        """
        Returns the total communication cost (in bytes) of the One-Shot Federated Training phase.
        Cost = Sum of (Serialized Size of Client Models).
        In a real scenario, this is the cost of uploading models to the server.
        """
        total_bytes = 0
        for c in self.components:
            # If component is LocalSPNWrapper or FederatedProduct, it has get_size_bytes
            if hasattr(c, "get_size_bytes"):
                total_bytes += c.get_size_bytes()
            else:
                # Fallback for generic nn.Module
                buffer = io.BytesIO()
                torch.save(c.state_dict(), buffer)
                total_bytes += buffer.tell()
        return total_bytes


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
