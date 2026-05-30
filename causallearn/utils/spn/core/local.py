"""
Local SPN implementations for federated learning.

This module contains the core local SPN wrappers used by clients.
"""

import io
import logging
import numpy as np
import torch
import torch.nn as nn
from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.normal import Normal


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
        structure="poon-domingos",  # BUG FIX: Accept structure parameter
    ):
        super().__init__()
        self.device = device
        self.mean = None
        self.std = None
        self.seed = seed
        self.num_features = num_features

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

        # BUG FIX: Ensure structural heterogeneity by seeding BEFORE model creation
        # This seeds the structure generation, not just training
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # BUG FIX: Use structure parameter from template instead of hardcoded "top-down"
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
            structure=structure,  # BUG FIX: Use passed structure parameter
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

    def train_local(
        self,
        data,
        weights=None,
        epochs=50,
        lr=0.005,
        l1_weight=1e-4,
        l2_weight=1e-5,
        grad_clip_norm=5.0,
        dropout=0.0,
    ):
        """
        Train this specific leaf on its data slice with L1 Sparsity + L2 Regularization.
        weights: Optional [N] array of sample weights (for EM).
        l1_weight: Weight for the L1 sparsity penalty on sum weights.
        l2_weight: Weight for L2 regularization (weight decay) for stability.
        grad_clip_norm: Maximum gradient norm for clipping (prevents explosion).
        dropout: Dropout rate for regularization (0.0 = no dropout).
        """
        if len(data) < 5:
            return 0.0

        # Compute and store normalization stats
        if self.mean is None:
            logging.error(
                f"[LocalSPNWrapper.fit] Computing stats from data.shape={data.shape}"
            )
            self.mean = torch.tensor(data.mean(axis=0), dtype=torch.float32).to(
                self.device
            )
            self.std = torch.tensor(data.std(axis=0), dtype=torch.float32).to(
                self.device
            )
            logging.error(
                f"[LocalSPNWrapper.fit] Computed mean.shape={self.mean.shape}, std.shape={self.std.shape}"
            )

        data_t = torch.tensor(data, dtype=torch.float32).to(self.device)
        data_t = self._normalize(data_t)
        data_t = self._permute(data_t)  # Apply dependency-aware ordering

        if weights is not None:
            weights_t = torch.tensor(weights, dtype=torch.float32).to(self.device)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Apply dropout if specified
        dropout_layer = None
        if dropout > 0.0:
            dropout_layer = torch.nn.Dropout(p=dropout)

        # Extract cluster and client indices from seed (seed = h * 10 + k)
        cluster_id = self.seed // 10 if self.seed is not None else -1
        client_id = self.seed % 10 if self.seed is not None else -1

        final_ll = 0.0
        loss_history = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Apply dropout to data if specified (simpler than modifying SPN internals)
            if dropout_layer is not None:
                data_with_dropout = dropout_layer(data_t)
                ll = self.model(data_with_dropout)
            else:
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

            # L2 Regularization (Weight Decay) for all parameters
            l2_penalty = 0.0
            for param in self.model.parameters():
                if param.requires_grad:
                    l2_penalty += torch.norm(param, p=2)

            loss = nll_loss + l1_weight * l1_penalty + l2_weight * l2_penalty

            loss.backward()

            # Gradient Clipping to prevent exploding gradients
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=grad_clip_norm
                )

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
        """
        Compute log P(x) with support for NaN marginalization.

        NaN Marginalization (Phase 2):
            When x contains NaN values, we compute P(X_obs) by marginalizing:
                P(X_obs) = ∫ P(X_obs, X_miss) dX_miss

            Key insight: ∫ P(X_miss | X_obs) dX_miss = 1
            Therefore: log(∫ P(X_miss | X_obs) dX_miss) = log(1) = 0

            This means NaN dimensions contribute 0 to log-likelihood, which is
            automatically handled by the Jacobian correction (mask zeroes out missing dims).

        Algorithm:
            1. Normalize and permute input (preserve NaNs)
            2. Forward pass through Einet (Einet handles NaN internally via marginalization)
            3. Jacobian correction (only applied to observed dimensions via mask)

        Args:
            x (Tensor): [batch, d] feature matrix (may contain NaNs)

        Returns:
            ll (Tensor): [batch, 1] log P(X_obs) where X_obs are non-NaN dimensions

        Reference: User insight + Seng et al. (2025) marginalization approach
        """
        # BUGFIX: Handle case when ALL features are NaN
        # When marginalizing over all dimensions: P(∅) = ∫ P(X) dX = 1 → log(1) = 0
        mask = (~torch.isnan(x)).float()
        all_nan_mask = mask.sum(dim=1) == 0  # Rows where all features are NaN

        if all_nan_mask.any():
            # If any row has all NaN, return 0 (log probability of marginalizing everything)
            batch_size = x.shape[0]
            ll = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
            ll[all_nan_mask] = 0.0  # log(1) = 0 for full marginalization

            # For rows with at least one observed feature, compute normally
            if (~all_nan_mask).any():
                x_obs = x[~all_nan_mask]
                x_norm = self._normalize(x_obs)
                x_perm = self._permute(x_norm)
                ll_obs = self.model(x_perm)

                if self.std is not None:
                    mask_obs = (~torch.isnan(x_obs)).float()
                    log_sigma = torch.log(self.std + 1e-6)
                    log_det_jacobian = -(log_sigma * mask_obs).sum(dim=1, keepdim=True)
                    ll_obs = ll_obs + log_det_jacobian

                ll[~all_nan_mask] = ll_obs

            return ll

        # Normal case: at least one feature is observed in all rows
        # 1. Normalize and Permute
        x_norm = self._normalize(x)
        x_perm = self._permute(x_norm)

        # 2. Forward pass through Einet
        # Einet's marginalization: NaN dimensions are integrated out
        ll = self.model(x_perm)

        # 3. Log-Jacobian Correction
        if self.std is not None:
            # Jacobian is based on scale sigma.
            # Mask is based on ORIGINAL x (not permuted)
            # NaN marginalization: mask zeroes out missing dims → no Jacobian contribution
            log_sigma = torch.log(self.std + 1e-6)
            log_det_jacobian = -(log_sigma * mask).sum(dim=1, keepdim=True)
            ll = ll + log_det_jacobian
        return ll

    def eval_partial_scope_log_likelihood(self, data, scope_indices):
        """
        Evaluate log P(X_scope) by marginalizing over unobserved variables.

        This is the critical method for CI testing. It computes the probability
        of a subset of variables by setting all others to NaN (marginalization).

        Args:
            data: Full data matrix [n, d]
            scope_indices: List of variable indices to compute probability for

        Returns:
            log_prob: [n, 1] log P(X_scope) where X_scope = {X_i : i in scope_indices}

        Example:
            >>> # Compute P(X_0, X_2) by marginalizing X_1, X_3, ..., X_d
            >>> ll = spn.eval_partial_scope_log_likelihood(data, [0, 2])
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)

        # Create masked data: keep only scope_indices, set others to NaN
        masked_data = torch.full_like(data, float("nan"))
        masked_data[:, scope_indices] = data[:, scope_indices]

        # Compute log P(X_scope) via NaN marginalization
        log_prob = self.log_prob(masked_data)

        return log_prob


class LocalClusterMixture(nn.Module):
    """
    Local mixture over K_local cluster SPNs for a single client.

    This represents one client's learned model after local clustering and training.
    Analogous to Seng's _build_cluster_mixture() in client.py.

    Mathematical Form:
        P_k(X) = Σ_{h=1}^{K_local} w_{k,h} × SPN_{k,h}(X)

    Args:
        cluster_spns (List[LocalSPNWrapper]): K_local SPNs for local clusters
        cluster_weights (np.ndarray): [K_local] weights (cluster sizes / n_k)
        client_id (int): Client identifier (for logging)
        device (str): 'cpu', 'cuda', or 'mps'

    Example:
        >>> # Client 0 with 400 samples, 2 local clusters
        >>> spn_0 = LocalSPNWrapper(...)  # Trained on 200 samples (cluster 0)
        >>> spn_1 = LocalSPNWrapper(...)  # Trained on 200 samples (cluster 1)
        >>> weights = [0.5, 0.5]  # Equal cluster sizes
        >>> mixture = LocalClusterMixture([spn_0, spn_1], weights, client_id=0)
        >>>
        >>> # Evaluation
        >>> x = torch.randn(100, 10)
        >>> log_p = mixture.log_prob(x)  # [100, 1]

    Reference:
        Seng et al. (2025), Algorithm 1, Lines 13-14
        - Line 13: cluster_local_data(OS(j), K)
        - Line 14: Learn dedicated PC for each cluster
    """

    def __init__(self, cluster_spns, cluster_weights, client_id=None, device="cpu"):
        super().__init__()

        self.client_id = client_id
        self.device = device
        self.K_local = len(cluster_spns)

        # Store cluster SPNs as ModuleList for proper PyTorch registration
        self.cluster_spns = nn.ModuleList(cluster_spns)

        # Convert weights to tensor
        if isinstance(cluster_weights, np.ndarray):
            self.weights = torch.tensor(cluster_weights, dtype=torch.float32).to(device)
        else:
            self.weights = torch.tensor(list(cluster_weights), dtype=torch.float32).to(
                device
            )

        # Validation
        assert len(self.cluster_spns) > 0, "Must have at least one cluster SPN"
        assert len(self.cluster_spns) == len(
            self.weights
        ), f"Mismatched SPNs ({len(self.cluster_spns)}) and weights ({len(self.weights)})"
        assert (
            abs(self.weights.sum().item() - 1.0) < 1e-5
        ), f"Weights must sum to 1, got {self.weights.sum().item()}"

        logging.info(
            f"[Client {client_id}] LocalClusterMixture created: "
            f"K_local={self.K_local}, weights={self.weights.cpu().numpy()}"
        )

    def log_prob(self, x):
        """
        Compute log P_k(X) = log(Σ_h w_{k,h} × SPN_{k,h}(X)).

        Algorithm:
            1. For each local cluster h: compute log P_{k,h}(x)
            2. Compute log(Σ_h w_h × P_h) via logsumexp trick

        Args:
            x (Tensor): [batch, d] input data

        Returns:
            log_prob (Tensor): [batch, 1] log probabilities
        """
        # Ensure tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)

        # BUGFIX: Handle case when ALL features are NaN
        # When marginalizing over all dimensions: P(∅) = 1 → log(1) = 0
        mask = (~torch.isnan(x)).float()
        all_nan_mask = mask.sum(dim=1) == 0

        if all_nan_mask.any():
            # Return 0 for rows with all NaN (full marginalization)
            batch_size = x.shape[0]
            log_prob = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)

            # For rows with at least one observed feature, compute normally
            if (~all_nan_mask).any():
                x_obs = x[~all_nan_mask]

                cluster_lls = []
                for spn in self.cluster_spns:
                    # Hybrid mode fix: Check if SPN expects different dimensions
                    spn_input = x_obs
                    if hasattr(spn, "mean") and spn.mean is not None:
                        expected_dims = spn.mean.shape[0]
                        if expected_dims > x_obs.shape[1]:
                            # Pad with NaN to marginalize
                            padding = torch.full(
                                (x_obs.shape[0], expected_dims - x_obs.shape[1]),
                                float("nan"),
                                device=x_obs.device,
                                dtype=x_obs.dtype,
                            )
                            spn_input = torch.cat([x_obs, padding], dim=1)

                    ll = spn.log_prob(spn_input)
                    cluster_lls.append(ll)

                ll_stack = torch.cat(cluster_lls, dim=1)
                log_weights = torch.log(self.weights + 1e-9).unsqueeze(0)
                log_prob_obs = torch.logsumexp(
                    ll_stack + log_weights, dim=1, keepdim=True
                )

                log_prob[~all_nan_mask] = log_prob_obs

            return log_prob

        # Normal case: at least one feature observed in all rows
        # Compute log-prob from each local cluster SPN
        cluster_lls = []
        for spn in self.cluster_spns:
            ll = spn.log_prob(x)  # [batch, 1]
            cluster_lls.append(ll)

        # Stack: [batch, K_local]
        ll_stack = torch.cat(cluster_lls, dim=1)

        # Log weights: [1, K_local]
        log_weights = torch.log(self.weights + 1e-9).unsqueeze(0)

        # Mixture via logsumexp: log(Σ_h w_h × P_h)
        log_prob = torch.logsumexp(ll_stack + log_weights, dim=1, keepdim=True)

        return log_prob  # [batch, 1]

    def sample(self, n):
        """
        Sample n data points from the local mixture.

        Algorithm:
            1. For each sample: choose cluster h ~ Categorical(weights)
            2. Sample from chosen cluster: x_i ~ SPN_{k,h}

        Args:
            n (int): Number of samples

        Returns:
            samples (Tensor): [n, d] generated samples
        """
        if n <= 0:
            return torch.tensor([], device=self.device)

        # Step 1: Choose which cluster to sample from for each data point
        cluster_indices = torch.multinomial(self.weights, n, replacement=True)  # [n]

        # Count samples per cluster for efficient batching
        unique_clusters, counts = torch.unique(cluster_indices, return_counts=True)

        # Step 2: Sample from each cluster
        samples_list = []
        for cluster_idx, count in zip(unique_clusters, counts):
            spn = self.cluster_spns[cluster_idx.item()]
            cluster_samples = spn.sample(count.item())  # [count, d]
            samples_list.append(cluster_samples)

        # Concatenate and shuffle
        all_samples = torch.cat(samples_list, dim=0)  # [n, d]
        perm = torch.randperm(all_samples.size(0))
        return all_samples[perm]

    def get_cluster_log_probs(self, x):
        """
        Get log probabilities from each cluster-specific SPN separately.

        Required for FederatedProductWithClusters to compute cluster-conditional densities.

        Args:
            x: Client's local data [batch_size, d_client]

        Returns:
            cluster_lls: [num_clusters, batch_size]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)

        cluster_lls = []
        for cluster_spn in self.cluster_spns:
            ll = cluster_spn.log_prob(x).squeeze()  # [batch_size]
            cluster_lls.append(ll)

        return torch.stack(cluster_lls, dim=0)  # [num_clusters, batch_size]

    def sample_from_clusters(self, cluster_assignments):
        """
        Sample from specific clusters.

        Required for FederatedProductWithClusters to sample conditioned on cluster L.

        Args:
            cluster_assignments: Array of cluster indices [n]

        Returns:
            samples: [n, d_client]
        """
        n = len(cluster_assignments)
        samples_list = []

        for cluster_id in range(self.K_local):
            # Find samples assigned to this cluster
            mask = cluster_assignments == cluster_id
            n_cluster = mask.sum()

            if n_cluster > 0:
                # Sample from cluster-specific SPN
                cluster_samples = self.cluster_spns[cluster_id].sample(n_cluster)
                samples_list.append((mask, cluster_samples))

        # Assemble full sample array
        # Get number of features from first sample
        if len(samples_list) > 0:
            n_features = samples_list[0][1].shape[1]
        else:
            # Fallback: sample one to get shape
            test_sample = self.cluster_spns[0].sample(1)
            n_features = test_sample.shape[1]

        samples = torch.zeros(n, n_features, device=self.device)
        for mask, cluster_samples in samples_list:
            samples[mask] = cluster_samples

        return samples

    def get_size_bytes(self):
        """
        Estimate memory footprint (sum of cluster SPNs).

        Returns:
            size_bytes (int): Total size in bytes
        """
        total_bytes = 0
        for spn in self.cluster_spns:
            if hasattr(spn, "get_size_bytes"):
                total_bytes += spn.get_size_bytes()
            else:
                buffer = io.BytesIO()
                torch.save(spn.state_dict(), buffer)
                total_bytes += buffer.tell()
        return total_bytes
