"""
Univariate SPN implementation using Gaussian Mixture Model.

This module provides a simple GMM-based SPN for 1-dimensional features.
"""

import io
import logging
import numpy as np
import torch
import torch.nn as nn


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
