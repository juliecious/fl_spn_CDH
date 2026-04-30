import io
import logging
from typing import Dict, List, Any

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


class LocalClusterMixture(nn.Module):
    """
    Represents a client's local mixture of cluster SPNs: P_k(X) = Σ_h w_{k,h} × SPN_{k,h}(X)

    This class implements Seng et al. (2025) client.py:383-397's local clustering approach.

    Mathematical Form:
        P_k(X) = Σ_{h=1}^{H_k} w_{k,h} × SPN_{k,h}(X)

    Where:
        - k: client index
        - H_k: number of LOCAL clusters for client k
        - w_{k,h}: mixture weight for cluster h in client k
        - SPN_{k,h}: SPN trained on client k's cluster h data

    Key Design Principles:
        1. LOCAL clustering: K-means runs ONLY on this client's data
        2. Sufficient data: Each cluster gets n_k / H_k samples (e.g., 400/2 = 200)
        3. Foundation for H/V/Hy modes: This mixture becomes a child node in global structure

    Reference: Seng et al. (2025), client.py lines 343-397 (_train_learned method)

    Args:
        cluster_spns (List[nn.Module]): H_k SPNs, one per local cluster
        cluster_weights (np.ndarray or List[float]): [H_k] mixture weights (must sum to 1)
        client_id (int): Client identifier (for logging/debugging)
        device (str): 'cpu', 'cuda', or 'mps'

    Example:
        >>> # Client 0 has 400 samples, clusters locally into H=2
        >>> spn_h0 = LocalSPNWrapper(num_features=8, device='cpu')
        >>> spn_h1 = LocalSPNWrapper(num_features=8, device='cpu')
        >>> # Train on 200 samples each...
        >>> mixture = LocalClusterMixture([spn_h0, spn_h1], cluster_weights=[0.5, 0.5],
        ...                               client_id=0, device='cpu')
        >>> x = torch.randn(100, 8)
        >>> log_p = mixture.log_prob(x)  # P_k(X) = 0.5 × P_{k,0}(X) + 0.5 × P_{k,1}(X)
    """

    def __init__(self, cluster_spns, cluster_weights, client_id, device="cpu"):
        super().__init__()

        # Store cluster SPNs as ModuleList for proper PyTorch registration
        self.cluster_spns = nn.ModuleList(cluster_spns)

        # Convert weights to tensor
        if isinstance(cluster_weights, np.ndarray):
            self.weights = torch.tensor(cluster_weights, dtype=torch.float32).to(device)
        else:
            self.weights = torch.tensor(list(cluster_weights), dtype=torch.float32).to(
                device
            )

        self.client_id = client_id
        self.device = device

        # Validation checks
        assert len(self.cluster_spns) > 0, "Must have at least one cluster SPN"
        assert len(self.cluster_spns) == len(
            self.weights
        ), f"Mismatched SPNs ({len(self.cluster_spns)}) and weights ({len(self.weights)})"
        assert (
            abs(self.weights.sum().item() - 1.0) < 1e-5
        ), f"Weights must sum to 1, got {self.weights.sum().item()}"

    def log_prob(self, x):
        """
        Compute log P_k(X) = log Σ_h [w_{k,h} × SPN_{k,h}(X)]

        Algorithm:
            1. For each cluster h: compute log SPN_{k,h}(X)
            2. Compute log Σ_h [w_{k,h} × SPN_{k,h}(X)] via logsumexp trick

        Mathematical Detail:
            log Σ_h [w_h × P_h] = logsumexp_h(log w_h + log P_h)

        Args:
            x (Tensor): [batch, d] feature matrix (may contain NaNs for marginalization)

        Returns:
            log_prob (Tensor): [batch, 1] log probabilities

        Reference: Seng et al. (2025), client.py:383-397
        """
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
                    ll = spn.log_prob(x_obs)
                    cluster_lls.append(ll)

                ll_stack = torch.cat(cluster_lls, dim=1)
                log_weights = torch.log(self.weights + 1e-9).unsqueeze(0)
                log_prob_obs = torch.logsumexp(
                    ll_stack + log_weights, dim=1, keepdim=True
                )

                log_prob[~all_nan_mask] = log_prob_obs

            return log_prob

        # Normal case: at least one feature observed in all rows
        # Step 1: Compute log-likelihoods from each cluster SPN
        cluster_lls = []
        for spn in self.cluster_spns:
            ll = spn.log_prob(x)  # [batch, 1]
            cluster_lls.append(ll)

        # Step 2: Stack and compute weighted mixture via logsumexp
        ll_stack = torch.cat(cluster_lls, dim=1)  # [batch, H_k]
        log_weights = torch.log(self.weights + 1e-9).unsqueeze(0)  # [1, H_k]

        # log Σ_h [w_h × P_h] = logsumexp(log w_h + log P_h)
        log_prob = torch.logsumexp(ll_stack + log_weights, dim=1, keepdim=True)

        return log_prob  # [batch, 1]

    def sample(self, n):
        """
        Sample n data points from the local cluster mixture.

        Algorithm:
            1. For each sample: choose cluster h ~ Categorical(weights)
            2. Sample from chosen cluster: x_i ~ SPN_{k,h}(X)

        Args:
            n (int): Number of samples to generate

        Returns:
            samples (Tensor): [n, d] samples from the mixture

        Reference: Ancestral sampling in mixture models
        """
        if n <= 0:
            return torch.tensor([], device=self.device)

        # Step 1: Choose which cluster to sample from for each data point
        comp_indices = torch.multinomial(self.weights, n, replacement=True)  # [n]

        # Count samples per cluster for efficiency
        unique_comps, counts = torch.unique(comp_indices, return_counts=True)

        # Step 2: Sample from each cluster and concatenate
        samples_list = []
        for comp_idx, count in zip(unique_comps, counts):
            spn = self.cluster_spns[comp_idx.item()]
            comp_samples = spn.sample(count.item())  # [count, d]

            # Ensure 2D shape
            if comp_samples.ndim == 1:
                # Infer d from first SPN's num_features
                num_features = self.cluster_spns[0].config.num_features
                comp_samples = comp_samples.view(-1, num_features)

            samples_list.append(comp_samples)

        # Concatenate all samples
        samples = torch.cat(samples_list, dim=0)  # [n, d]

        return samples

    def get_size_bytes(self):
        """
        Estimate memory footprint of this local cluster mixture.

        Returns:
            int: Total size in bytes (sum of all cluster SPNs)
        """
        total_size = 0
        for spn in self.cluster_spns:
            buffer = io.BytesIO()
            torch.save(spn.state_dict(), buffer)
            total_size += buffer.tell()
        return total_size


class EnsembleSPNWrapper(nn.Module):
    """
    Ensemble of multiple LocalSPNWrapper models with adaptive architecture scaling.

    Combines two optimizations:
    1. Adaptive Scaling: Increase capacity with dimensionality
    2. Ensemble: Multiple models with different random seeds → lower variance

    Expected benefits:
    - Reduces bias (Option 1): Better individual model accuracy
    - Reduces variance (Option 2): More stable CI test estimates
    - Synergy: Better models → better ensemble

    Usage:
        ensemble = EnsembleSPNWrapper(num_features=8, n_models=5, device='cpu')
        ensemble.train_local(X_train, epochs=50)
        ll = ensemble.log_prob(X_test)
    """

    def __init__(
        self,
        num_features,
        device="cpu",
        n_models=5,
        base_sums=20,
        base_leaves=20,
        base_reps=10,
        seed=42,
        variable_order=None,
    ):
        """
        Initialize ensemble with adaptive architecture scaling.

        Args:
            num_features: Number of features
            device: 'cpu', 'cuda', or 'mps'
            n_models: Number of models in ensemble (default: 5)
            base_sums: Base num_sums before scaling (default: 20)
            base_leaves: Base num_leaves before scaling (default: 20)
            base_reps: Base num_repetitions before scaling (default: 10)
            seed: Base random seed (each model gets seed+i)
            variable_order: Optional dependency-aware variable ordering
        """
        super().__init__()
        self.num_features = num_features
        self.device = device
        self.n_models = n_models

        # Option 1: Adaptive architecture scaling
        # Scale capacity with dimensionality
        num_sums = base_sums + num_features * 2
        num_leaves = base_leaves + num_features * 2
        num_repetitions = base_reps + num_features // 2

        # Depth constraint: 2^depth <= num_features
        depth = int(np.floor(np.log2(num_features)))

        logging.info(
            f"EnsembleSPN: {n_models} models with scaled architecture "
            f"(sums={num_sums}, leaves={num_leaves}, reps={num_repetitions}, depth={depth})"
        )

        # Option 2: Create ensemble of models with different seeds
        self.models = nn.ModuleList()
        for i in range(n_models):
            model = LocalSPNWrapper(
                num_features=num_features,
                device=device,
                depth=depth,
                num_sums=num_sums,
                num_leaves=num_leaves,
                num_repetitions=num_repetitions,
                seed=seed + i,  # Different seed for each model
                variable_order=variable_order,
            )
            self.models.append(model)

        # Store mean/std from first model (all will compute same normalization)
        self.mean = None
        self.std = None

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
        Train all models in ensemble.

        Can be parallelized if needed, but sequential is fine for CPU.
        """
        if len(data) < 5:
            return 0.0

        final_lls = []
        for i, model in enumerate(self.models):
            # Train each model
            ll = model.train_local(
                data,
                weights=weights,
                epochs=epochs,
                lr=lr,
                l1_weight=l1_weight,
                l2_weight=l2_weight,
                grad_clip_norm=grad_clip_norm,
                dropout=dropout,
            )
            final_lls.append(ll)

            # Store normalization from first model
            if i == 0 and self.mean is None:
                self.mean = model.mean
                self.std = model.std

        # Return average final LL
        avg_ll = np.mean(final_lls)
        logging.info(
            f"  Ensemble trained: avg final LL = {avg_ll:.4f} "
            f"(range: [{min(final_lls):.4f}, {max(final_lls):.4f}])"
        )
        return avg_ll

    def log_prob(self, x):
        """
        Compute ensemble log probability: log(1/K * Σ_k exp(ll_k))

        Uses logsumexp for numerical stability.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Collect log-probs from all models
        lls = []
        for model in self.models:
            ll = model.log_prob(x)
            lls.append(ll)

        # Stack: [n_models, batch_size, 1]
        lls_stacked = torch.stack(lls, dim=0)

        # Average in log-space: log(1/K * Σ exp(ll_k)) = logsumexp(ll_k) - log(K)
        ensemble_ll = torch.logsumexp(lls_stacked, dim=0) - np.log(self.n_models)

        return ensemble_ll

    def sample(self, n):
        """
        Sample from ensemble by randomly selecting a model for each sample.
        """
        samples_list = []
        samples_per_model = n // self.n_models
        remainder = n % self.n_models

        for i, model in enumerate(self.models):
            n_samples = samples_per_model + (1 if i < remainder else 0)
            if n_samples > 0:
                samples = model.sample(n_samples)
                samples_list.append(samples)

        # Concatenate and shuffle
        all_samples = torch.cat(samples_list, dim=0)
        perm = torch.randperm(all_samples.size(0))
        return all_samples[perm]

    def ll(self, x):
        """Alias for log_prob for compatibility."""
        return self.log_prob(x).squeeze()


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


class GroupMixture(nn.Module):
    """
    Mixture of client SPNs for a specific feature group.

    Mathematical Form:
        P(X_g) = Σ_k w_k,g × P_k,g(X_g)

    Reference: Seng et al. (2025), Section 3.2 "Sum Nodes and Horizontal FL"

    This class represents a Sum node in the Mixture-then-Product hierarchy.
    It models a single feature group (subspace) as a mixture over K clients
    that have trained SPNs on that feature group.

    Design Rationale:
        - Handles overlapping features: Same feature group can appear in multiple clients
        - Weights represent client contributions (typically proportional to sample counts)
        - Essential building block for hybrid federated learning

    Args:
        client_spns (List[nn.Module]): K SPNs from different clients, each trained on X_g
        weights (np.ndarray or List[float]): [K] mixture weights, must sum to 1
        feature_indices (List[int]): Which features (columns) this group models
        device (str): 'cpu' or 'cuda'
        full_d (int, optional): If SPNs are trained on full dimensionality (not just subspace),
            pass full_d to enable NaN masking. Default None (SPNs match feature_indices)

    Example:
        >>> # Two clients share features [0, 1, 2]
        >>> spn0 = LocalSPNWrapper(num_features=3, device='cpu')
        >>> spn1 = LocalSPNWrapper(num_features=3, device='cpu')
        >>> # Train SPNs on local data...
        >>> mixture = GroupMixture([spn0, spn1], weights=[0.4, 0.6],
        ...                        feature_indices=[0,1,2], device='cpu')
        >>> x = torch.randn(100, 5)  # 100 samples, 5 total features
        >>> log_p = mixture.log_prob(x)  # Extracts x[:, [0,1,2]] internally
    """

    def __init__(
        self, client_spns, weights, feature_indices, device="cpu", full_d=None
    ):
        super().__init__()

        # Store client SPNs as ModuleList for proper PyTorch registration
        self.client_spns = nn.ModuleList(client_spns)

        # Convert weights to tensor
        if isinstance(weights, np.ndarray):
            self.weights = torch.tensor(weights, dtype=torch.float32).to(device)
        else:
            self.weights = torch.tensor(list(weights), dtype=torch.float32).to(device)

        # Store feature indices and device
        self.feature_indices = list(feature_indices)
        self.device = device
        self.full_d = full_d  # If not None, SPNs expect full_d dimensions

        # Performance optimization: Pre-compute log weights (called in every forward pass)
        self.log_weights = torch.log(self.weights + 1e-9).unsqueeze(0)  # [1, K]

        # Performance optimization: Pre-compute feature mask for NaN tensor creation
        # This avoids creating the mask on every forward pass
        if self.full_d is not None:
            # Create a boolean mask for features NOT in this group
            self.nan_mask = torch.ones(self.full_d, dtype=torch.bool, device=device)
            self.nan_mask[self.feature_indices] = False
        else:
            self.nan_mask = None

        # Validation checks
        assert len(self.client_spns) > 0, "Must have at least one client SPN"
        assert len(self.client_spns) == len(
            self.weights
        ), f"Mismatched SPNs ({len(self.client_spns)}) and weights ({len(self.weights)})"
        assert (
            abs(self.weights.sum().item() - 1.0) < 1e-5
        ), f"Weights must sum to 1, got {self.weights.sum().item()}"
        assert len(self.feature_indices) > 0, "Feature indices cannot be empty"

    def log_prob(self, x):
        """
        Compute log P(X_g) where X_g are the features in this group.

        NaN Handling (Phase 3):
            NaN marginalization is delegated to child SPNs (LocalSPNWrapper or LocalClusterMixture).
            Each client's SPN handles NaN dimensions via marginalization (Phase 2).
            This class simply aggregates the marginalized log-likelihoods via mixture.

        Algorithm:
            1. Extract features: x_g = x[:, feature_indices] (may contain NaNs)
            2. For each client k: compute log P_k(x_g) (NaN handled by client SPN)
            3. Compute log Σ_k [w_k × P_k(x_g)] via logsumexp trick

        Mathematical Detail:
            log Σ_k [w_k × P_k] = logsumexp_k(log w_k + log P_k)

        Args:
            x (Tensor): [batch, d_full] full feature matrix (may contain NaNs)

        Returns:
            log_prob (Tensor): [batch, 1] log probabilities for this group

        Reference: Seng et al. (2025), Definition 1 (Horizontal FL mixture)
        """
        # BUGFIX: Handle context column if present
        # In hybrid mode with horizontal-style context column, input may be [batch, d+1]
        # where last column is context. Strip it before processing.
        if self.full_d is not None and x.shape[1] > self.full_d:
            # Context column detected (e.g., x is [batch, 9] but full_d=8)
            x = x[:, : self.full_d]  # Strip context column → [batch, 8]

        # Step 1: Extract features for this group
        # Justification: Each group only models its subset of features
        # If full_d is set, use NaN masking instead of extraction
        if self.full_d is not None:
            # Performance optimization: Clone and mask instead of full + copy
            # This is faster for small feature groups (typical case)
            x_g = x.clone()  # [batch, d]
            x_g[:, self.nan_mask] = float("nan")  # Mask features not in this group
        else:
            # SPNs trained on subspace - extract features
            x_g = x[:, self.feature_indices]  # [batch, len(feature_indices)]

        # BUGFIX: Handle case when ALL features in this group are NaN
        # When marginalizing over all dimensions in this group: P(∅) = 1 → log(1) = 0
        mask = (~torch.isnan(x_g)).float()
        all_nan_mask = mask.sum(dim=1) == 0

        if all_nan_mask.any():
            # Return 0 for rows with all NaN (full marginalization for this group)
            batch_size = x_g.shape[0]
            log_prob = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)

            # For rows with at least one observed feature in this group, compute normally
            if (~all_nan_mask).any():
                x_g_obs = x_g[~all_nan_mask]

                # Performance: Pre-allocate tensor instead of list append
                ll_stack = torch.empty(
                    x_g_obs.shape[0],
                    len(self.client_spns),
                    device=x.device,
                    dtype=x.dtype,
                )
                for i, spn in enumerate(self.client_spns):
                    ll_stack[:, i : i + 1] = spn.log_prob(x_g_obs)  # [batch_obs, 1]

                # Use pre-computed log_weights
                log_prob_obs = torch.logsumexp(
                    ll_stack + self.log_weights, dim=1, keepdim=True
                )

                log_prob[~all_nan_mask] = log_prob_obs

            return log_prob

        # Step 2: Compute log-likelihoods from each client SPN
        # Justification: Each client contributes its learned distribution
        # NaN marginalization: Delegated to client SPNs (Phase 2)
        # Performance: Pre-allocate tensor instead of list append + cat
        ll_stack = torch.empty(
            x_g.shape[0], len(self.client_spns), device=x.device, dtype=x.dtype
        )
        for i, spn in enumerate(self.client_spns):
            ll_stack[:, i : i + 1] = spn.log_prob(x_g)  # [batch, 1]

        # Step 3: Stack and compute weighted mixture via logsumexp
        # Justification: Numerically stable computation of log(Σ exp(...))
        # Performance: Use pre-computed log_weights from __init__
        # log Σ_k [w_k × P_k] = logsumexp(log w_k + log P_k)
        log_prob = torch.logsumexp(ll_stack + self.log_weights, dim=1, keepdim=True)

        return log_prob  # [batch, 1]

    def sample(self, n):
        """
        Sample n data points from the mixture distribution.

        Algorithm:
            1. For each sample: choose client k ~ Categorical(weights)
            2. Sample from chosen client: x_i ~ P_k(X_g)
            3. Return samples for this feature group only

        Args:
            n (int): Number of samples to generate

        Returns:
            samples (Tensor): [n, len(feature_indices)] samples for this group

        Note: Returns samples ONLY for this group's features, not full d-dimensional.
              Caller (ProductOverGroups) will concatenate group samples.
        """
        if n <= 0:
            return torch.tensor([], device=self.device)

        # Step 1: Choose which client to sample from for each data point
        # Justification: Ancestral sampling - choose mixture component first
        comp_indices = torch.multinomial(self.weights, n, replacement=True)  # [n]

        # Count samples per client for efficiency
        unique_comps, counts = torch.unique(comp_indices, return_counts=True)

        # Step 2: Sample from each client and concatenate
        # Justification: Batch sampling is more efficient than individual samples
        samples_list = []
        for comp_idx, count in zip(unique_comps, counts):
            spn = self.client_spns[comp_idx.item()]
            comp_samples = spn.sample(
                count.item()
            )  # May be [count, d_full] if full SPN

            # BUGFIX: If SPN returns full-dimensional samples, extract only relevant features
            # This happens in vertical mode where SPNs are trained on feature subsets
            # but still represent full d-dimensional distribution
            if self.full_d is not None and comp_samples.shape[1] > len(
                self.feature_indices
            ):
                # Extract only the features for this group
                comp_samples = comp_samples[
                    :, self.feature_indices
                ]  # [count, len(feature_indices)]

            # Ensure 2D shape
            if comp_samples.ndim == 1:
                comp_samples = comp_samples.view(-1, len(self.feature_indices))

            samples_list.append(comp_samples)

        # Concatenate all samples
        samples = torch.cat(samples_list, dim=0)  # [n, len(feature_indices)]

        return samples

    def get_size_bytes(self):
        """
        Estimate memory footprint of this mixture (sum of client SPNs).

        Returns:
            size_bytes (int): Total size in bytes
        """
        total_bytes = 0
        for spn in self.client_spns:
            if hasattr(spn, "get_size_bytes"):
                total_bytes += spn.get_size_bytes()
            else:
                # Fallback: serialize to estimate size
                buffer = io.BytesIO()
                torch.save(spn.state_dict(), buffer)
                total_bytes += buffer.tell()
        return total_bytes


class ProductOverGroups(nn.Module):
    """
    Product of feature group mixtures (disjoint groups only).

    Mathematical Form:
        P(X) = Π_g P(X_g)  where feature groups are disjoint
        log P(X) = Σ_g log P(X_g)

    Reference: Seng et al. (2025), Section 3.2 "Product Nodes and Vertical FL"
               Definition 2 (Vertical Federated Learning)

    Design Rationale:
        - Combines multiple GroupMixtures via product operation
        - Assumes conditional independence: P(X_g1, X_g2 | mixture) = P(X_g1) × P(X_g2)
        - Critical building block for hybrid federated learning
        - Enforces disjoint constraint (overlaps require ProductOverGroupsWithOverlap)

    Args:
        group_mixtures (List[GroupMixture]): G mixtures, one per feature group
        feature_groups (List[List[int]]): Feature indices for each group
        device (str): 'cpu' or 'cuda'

    Raises:
        ValueError: If feature groups overlap (use ProductOverGroupsWithOverlap)
        AssertionError: If group_mixtures and feature_groups counts mismatch

    Example:
        >>> # Two disjoint groups: [0,1,2] and [3,4]
        >>> mix1 = GroupMixture([spn0, spn1], [0.5, 0.5], feature_indices=[0,1,2])
        >>> mix2 = GroupMixture([spn2, spn3], [0.5, 0.5], feature_indices=[3,4])
        >>> product = ProductOverGroups([mix1, mix2], [[0,1,2], [3,4]], device='cpu')
        >>> x = torch.randn(100, 5)
        >>> log_p = product.log_prob(x)  # log P(X) = log P(X_012) + log P(X_34)
    """

    def __init__(self, group_mixtures, feature_groups, device="cpu"):
        super().__init__()

        # Store group mixtures as ModuleList for proper PyTorch registration
        # Justification: Same as GroupMixture - ensures parameter tracking
        self.group_mixtures = nn.ModuleList(group_mixtures)

        # Store feature groups and device
        # Justification: Need to know which features each group models for sampling
        self.feature_groups = [list(group) for group in feature_groups]
        self.device = device

        # Validation checks
        assert len(self.group_mixtures) > 0, "Must have at least one group mixture"
        assert len(self.group_mixtures) == len(self.feature_groups), (
            f"Mismatched group_mixtures ({len(self.group_mixtures)}) and "
            f"feature_groups ({len(self.feature_groups)})"
        )

        # Critical: Validate that feature groups are disjoint
        # Justification: Overlapping features violate product semantics
        # (would lead to incorrect probability - features counted multiple times)
        self._validate_disjoint()

        # Compute total dimensionality for sampling
        # Justification: Need to know output shape [n, d] for sample()
        all_indices = []
        for group in self.feature_groups:
            all_indices.extend(group)
        self.num_features = max(all_indices) + 1 if all_indices else 0

    def _validate_disjoint(self):
        """
        Validate that feature groups are disjoint (no overlaps).

        Justification:
            Product P(X) = Π_g P(X_g) is only valid when scopes are disjoint.
            If features overlap, we'd be multiplying probabilities that share variables,
            leading to incorrect normalization: ∫ P(X) dx ≠ 1

        Raises:
            ValueError: If any feature appears in multiple groups

        Example of INVALID input:
            feature_groups = [[0,1,2], [2,3,4]]  # Feature 2 appears twice!
        """
        all_features = []
        for group in self.feature_groups:
            all_features.extend(group)

        # Check for duplicates: len(list) > len(set) means duplicates exist
        if len(all_features) != len(set(all_features)):
            # Find which features are duplicated for error message
            from collections import Counter

            counts = Counter(all_features)
            duplicates = [f for f, c in counts.items() if c > 1]

            raise ValueError(
                f"Feature groups overlap! Features {duplicates} appear in multiple groups. "
                f"Use ProductOverGroupsWithOverlap for overlapping features. "
                f"Got groups: {self.feature_groups}"
            )

    def log_prob(self, x):
        """
        Compute log P(X) = Σ_g log P(X_g).

        Algorithm:
            1. For each group g: compute log P(X_g) via GroupMixture
            2. Sum all log probabilities (product in probability space)

        Mathematical Detail:
            P(X) = Π_g P(X_g)
            log P(X) = log Π_g P(X_g) = Σ_g log P(X_g)

        Justification:
            - Sum in log space is product in probability space
            - Each GroupMixture extracts its features internally via x[:, feature_indices]
            - No feature coordination needed (disjoint guarantee)

        Args:
            x (Tensor): [batch, d] full feature matrix

        Returns:
            log_prob (Tensor): [batch, 1] log probabilities

        Reference: Product node evaluation (Seng et al. 2025, Section 2)
        """
        # Compute log-prob for each group
        # Justification: Each GroupMixture independently evaluates its feature subspace
        group_lls = []
        for mixture_g in self.group_mixtures:
            ll_g = mixture_g.log_prob(x)  # [batch, 1]
            group_lls.append(ll_g)

        # Sum log-probs (equivalent to product in probability space)
        # Justification: log(a × b) = log(a) + log(b)
        ll_stack = torch.cat(group_lls, dim=1)  # [batch, G]
        total_ll = torch.sum(ll_stack, dim=1, keepdim=True)  # [batch, 1]

        return total_ll

    def sample(self, n):
        """
        Sample from product distribution by sampling each group independently.

        Algorithm:
            1. For each group g: sample x_g ~ P(X_g)
            2. Assemble samples into full d-dimensional feature vector

        Justification:
            Product distribution has independent groups, so we can sample each separately.
            This is ancestral sampling for product nodes: sample each child independently,
            then combine.

        Args:
            n (int): Number of samples to generate

        Returns:
            samples (Tensor): [n, d] where d = total number of features

        Reference: Product node sampling (Seng et al. 2025, implicit in Section 2)

        Example:
            If feature_groups = [[0,1,2], [3,4]], then:
            - Sample x[:, [0,1,2]] from mixture_0
            - Sample x[:, [3,4]] from mixture_1
            - Concatenate to form full x[:, 0:5]
        """
        if n <= 0:
            return torch.tensor([], device=self.device)

        # Initialize full sample matrix
        # Justification: Pre-allocate for efficiency, fill in groups
        samples = torch.zeros(n, self.num_features, device=self.device)

        # Sample each group independently
        # Justification: Product means groups are independent
        for g, mixture_g in enumerate(self.group_mixtures):
            # Sample from this group's mixture
            group_samples = mixture_g.sample(n)  # Should be [n, len(indices)]

            # Place samples in correct feature positions
            # Justification: Each group models specific features (stored in feature_groups)
            indices = self.feature_groups[g]

            # BUGFIX: Double-check dimensions and extract if needed (defense in depth)
            # This handles cases where GroupMixture.sample() doesn't properly extract features
            expected_cols = len(indices)
            if group_samples.ndim == 1:
                group_samples = group_samples.view(n, 1)

            actual_cols = group_samples.shape[1]

            if actual_cols != expected_cols:
                # Shape mismatch - try to fix it
                if actual_cols > expected_cols:
                    # Too many columns - extract only the ones we need
                    # This happens if GroupMixture returns full-dimensional samples
                    if actual_cols == self.num_features:
                        # Full-dimensional samples - extract using indices
                        group_samples = group_samples[:, indices]
                    else:
                        # Partial but wrong size - take first N columns as fallback
                        group_samples = group_samples[:, :expected_cols]
                else:
                    # Too few columns - this is a real error
                    raise ValueError(
                        f"ProductOverGroups.sample(): Group {g} returned {actual_cols} columns "
                        f"but expected {expected_cols} for indices {indices}. "
                        f"group_samples shape={group_samples.shape}, indices={indices}"
                    )

            # Validate dimensions one more time before assignment
            assert group_samples.shape == (
                n,
                expected_cols,
            ), f"Shape mismatch after extraction: {group_samples.shape} != ({n}, {expected_cols})"

            samples[:, indices] = group_samples

        return samples  # [n, d]

    def get_size_bytes(self):
        """
        Estimate memory footprint (sum of all group mixtures).

        Justification:
            Communication cost in federated learning = sum of component sizes.
            This matches the federated aggregation: each GroupMixture was sent from clients.

        Returns:
            size_bytes (int): Total size in bytes
        """
        total_bytes = 0
        for mixture in self.group_mixtures:
            if hasattr(mixture, "get_size_bytes"):
                total_bytes += mixture.get_size_bytes()
            else:
                # Fallback: serialize to estimate
                buffer = io.BytesIO()
                torch.save(mixture.state_dict(), buffer)
                total_bytes += buffer.tell()
        return total_bytes


class ProductOverGroupsWithOverlap(nn.Module):
    """
    Product over groups with overlapping feature support.

    Uses indicator matrix method from Seng et al. (2025), Algorithm 1.

    Mathematical Form:
        P(X) = Π_g P(X_g)  where each feature in exactly one group
        (groups auto-constructed to handle overlaps)

    Reference: Seng et al. (2025), Algorithm 1 "Building Structure via Data Partitioning"

    Key Design Insight:
        Overlap handling happens at CONSTRUCTION time, not inference time.

        Algorithm 1 from Seng et al.:
        1. Build indicator matrix M[k,j] = 1 if client k has feature j
        2. Group features by unique column patterns (client sets)
        3. Each group gets a mixture over clients that have those features
        4. Product combines groups (now disjoint by construction)

        Example:
            Client 0 has features [0, 1, 2]
            Client 1 has features [1, 2, 3]

            Automatic grouping:
                Group A: [0] (only client 0)      → SPN from client 0
                Group B: [1, 2] (both clients)    → Mixture of client 0 & 1
                Group C: [3] (only client 1)      → SPN from client 1

            Result: P(X) = P(X_0) × P(X_{1,2}) × P(X_3)
                         = P_0(X_0) × [0.5×P_0(X_{1,2}) + 0.5×P_1(X_{1,2})] × P_1(X_3)

        This PREVENTS double-counting: features [1,2] appear in exactly ONE group (B).

    Design Rationale:
        - Extends ProductOverGroups to handle overlapping features
        - Same inference logic as ProductOverGroups (overlap resolved at construction)
        - GroupMixtures are already constructed with correct feature scopes
        - Validates that construction was done correctly (no overlaps in final structure)

    Args:
        group_mixtures (List[GroupMixture]): G mixtures, correctly partitioned
        feature_groups (List[List[int]]): Feature indices for each group
        device (str): 'cpu' or 'cuda'
        allow_overlap (bool): If False, validates disjoint (default True for this class)

    Raises:
        ValueError: If feature_groups have overlaps when allow_overlap=False
        Warning: If allow_overlap=True but overlaps detected (construction error)

    Example:
        >>> # Construct GroupMixtures following Algorithm 1
        >>> mix_A = GroupMixture([spn_c0], [1.0], feature_indices=[0])
        >>> mix_B = GroupMixture([spn_c0, spn_c1], [0.5, 0.5], feature_indices=[1,2])
        >>> mix_C = GroupMixture([spn_c1], [1.0], feature_indices=[3])
        >>> product = ProductOverGroupsWithOverlap([mix_A, mix_B, mix_C],
        ...                                         [[0], [1,2], [3]])
        >>> # Now features [1,2] handled by single mixture (no double-counting)
    """

    def __init__(
        self, group_mixtures, feature_groups, device="cpu", allow_overlap=True
    ):
        super().__init__()

        # Store group mixtures as ModuleList
        # Justification: Same as ProductOverGroups - proper PyTorch registration
        self.group_mixtures = nn.ModuleList(group_mixtures)

        # Store feature groups and device
        self.feature_groups = [list(group) for group in feature_groups]
        self.device = device
        self.allow_overlap = allow_overlap

        # Validation
        assert len(self.group_mixtures) > 0, "Must have at least one group mixture"
        assert len(self.group_mixtures) == len(self.feature_groups), (
            f"Mismatched group_mixtures ({len(self.group_mixtures)}) and "
            f"feature_groups ({len(self.feature_groups)})"
        )

        # Detect overlaps
        # Justification: Even with allow_overlap=True, we want to know if overlaps exist
        # for diagnostic purposes and to warn about potential construction errors
        self.overlap_info = self._detect_overlaps()

        # If overlaps detected with allow_overlap=False, raise error
        # Justification: Caller claims structure is disjoint, but it's not
        if not allow_overlap and self.overlap_info["has_overlap"]:
            raise ValueError(
                f"Feature groups overlap, but allow_overlap=False. "
                f"Overlapping features: {self.overlap_info['overlapping_features']}. "
                f"If this is intentional, set allow_overlap=True."
            )

        # If overlaps detected with allow_overlap=True, log warning
        # Justification: Overlaps should have been resolved during construction (Algorithm 1)
        # If they still exist, it's likely a construction error
        if allow_overlap and self.overlap_info["has_overlap"]:
            import logging

            logging.warning(
                f"ProductOverGroupsWithOverlap: Feature groups have overlaps. "
                f"This is allowed but unusual - overlaps should be resolved at construction. "
                f"Overlapping features: {self.overlap_info['overlapping_features']}"
            )

        # Compute total dimensionality
        # Justification: Same as ProductOverGroups - needed for sampling
        all_indices = []
        for group in self.feature_groups:
            all_indices.extend(group)
        self.num_features = max(all_indices) + 1 if all_indices else 0

    def _detect_overlaps(self):
        """
        Detect which features appear in multiple groups.

        Justification:
            - Diagnostic tool to verify Algorithm 1 was applied correctly
            - If overlaps exist, either:
              a) Construction error (most likely)
              b) Intentional design (user's responsibility to ensure correctness)

        Returns:
            dict: {
                'has_overlap': bool,
                'overlapping_features': List[int],
                'feature_to_groups': Dict[int, List[int]]
            }

        Reference: Seng et al. (2025), Algorithm 1 implicit overlap detection
        """
        feature_to_groups = {}
        for g, group in enumerate(self.feature_groups):
            for feat in group:
                if feat not in feature_to_groups:
                    feature_to_groups[feat] = []
                feature_to_groups[feat].append(g)

        overlapping_features = [
            f for f, groups in feature_to_groups.items() if len(groups) > 1
        ]
        has_overlap = len(overlapping_features) > 0

        return {
            "has_overlap": has_overlap,
            "overlapping_features": overlapping_features,
            "feature_to_groups": feature_to_groups,
        }

    def log_prob(self, x):
        """
        Compute log P(X) = Σ_g log P(X_g).

        Mathematical Detail:
            If structure was built correctly per Algorithm 1, each feature appears
            in exactly one GroupMixture. Therefore, we can safely sum log-probs.

            P(X) = Π_g P(X_g)  (disjoint scopes after construction)
            log P(X) = Σ_g log P(X_g)

        Justification:
            - Same as ProductOverGroups because overlap resolved at construction
            - Each GroupMixture extracts its features via x[:, feature_indices]
            - No special overlap handling needed at inference time
            - This is the KEY INSIGHT from Seng et al. (2025): "resolve overlaps
              during structure building, not during inference"

        Args:
            x (Tensor): [batch, d] full feature matrix

        Returns:
            log_prob (Tensor): [batch, 1] log probabilities

        Reference: Seng et al. (2025), Section 3.2 "Product evaluation is simple
                   because structure ensures disjoint scopes"
        """
        # Compute log-prob for each group
        # Justification: Each GroupMixture independently evaluates its feature subspace
        group_lls = []
        for mixture_g in self.group_mixtures:
            ll_g = mixture_g.log_prob(x)  # [batch, 1]
            group_lls.append(ll_g)

        # Sum log-probs (product in probability space)
        # Justification: log(a × b) = log(a) + log(b)
        ll_stack = torch.cat(group_lls, dim=1)  # [batch, G]
        total_ll = torch.sum(ll_stack, dim=1, keepdim=True)  # [batch, 1]

        return total_ll

    def sample(self, n):
        """
        Sample from product distribution by sampling each group independently.

        Algorithm:
            1. For each group g: sample x_g ~ P(X_g)
            2. Assemble samples into full d-dimensional feature vector

        Justification:
            - Same as ProductOverGroups (overlap resolved at construction)
            - Product means groups are independent
            - Each GroupMixture samples its feature subspace
            - If overlaps exist, later groups overwrite earlier (but shouldn't happen
              if construction followed Algorithm 1)

        Args:
            n (int): Number of samples to generate

        Returns:
            samples (Tensor): [n, d] where d = total number of features

        Reference: Seng et al. (2025), ancestral sampling for product nodes
        """
        if n <= 0:
            return torch.tensor([], device=self.device)

        # Initialize full sample matrix
        samples = torch.zeros(n, self.num_features, device=self.device)

        # Sample each group independently
        # Justification: Product means groups are independent
        for g, mixture_g in enumerate(self.group_mixtures):
            group_samples = mixture_g.sample(n)  # [n, d_g]

            # Place samples in correct feature positions
            indices = self.feature_groups[g]
            samples[:, indices] = group_samples

        return samples  # [n, d]

    def get_size_bytes(self):
        """
        Estimate memory footprint (sum of all group mixtures).

        Returns:
            size_bytes (int): Total size in bytes
        """
        total_bytes = 0
        for mixture in self.group_mixtures:
            if hasattr(mixture, "get_size_bytes"):
                total_bytes += mixture.get_size_bytes()
            else:
                buffer = io.BytesIO()
                torch.save(mixture.state_dict(), buffer)
                total_bytes += buffer.tell()
        return total_bytes


class GlobalSumOfProducts(nn.Module):
    """
    Global sum over multiple product SPNs (sum-over-cluster-combinations).

    Implements Seng's critical structure: "sum nodes on top of the products
    that group these clusters" (Seng feedback, April 29, 2026).

    Mathematical Form:
        P(X) = Σ_c w_c × Product_c(X)
        where Product_c(X) = ∏_g P(X_g | cluster_config_c)

    Key Insight - Why This Breaks Independence:
        Without sum (current): P(X) = ∏_g P(X_g) → I(X_g1; X_g2) = 0 (always!)
        With sum (correct):    P(X) = Σ_c w_c × ∏_g P_c(X_g)
                              → I(X_g1; X_g2) ≠ 0 (can model dependencies!)

        The sum "couples" feature groups through shared cluster assignments:
        If X_g1 belongs to cluster A, X_g2 is more likely in cluster A too.

    Mathematical Proof of Dependency:
        P(X, Y) = w1×P(X|A)×P(Y|A) + w2×P(X|B)×P(Y|B)
        P(X) = w1×P(X|A) + w2×P(X|B)
        P(Y) = w1×P(Y|A) + w2×P(Y|B)

        P(X)×P(Y) = w1²P(X|A)P(Y|A) + w1w2[P(X|A)P(Y|B) + P(X|B)P(Y|A)] + w2²P(X|B)P(Y|B)

        This differs from P(X,Y) → I(X;Y) ≠ 0

    Analogy to Mixture of Gaussians:
        - Each Gaussian has diagonal covariance (assumes independence)
        - But mixture of Gaussians can model correlations!
        - Same principle: mixture of products can model dependencies

    Args:
        products (List[ProductOverGroupsWithOverlap]): Multiple product SPNs,
            each representing a different cluster configuration
        weights (List[float] or np.ndarray): Mixture weights (must sum to 1)
        device (str): 'cpu', 'cuda', or 'mps'

    Example:
        >>> # 3 cluster configurations for K=3 clients, K_local=2
        >>> # Config 1: (0,0,0) - all clients use cluster 0
        >>> prod1 = ProductOverGroupsWithOverlap(...)
        >>> # Config 2: (0,1,1) - client 0 uses cluster 0, others use cluster 1
        >>> prod2 = ProductOverGroupsWithOverlap(...)
        >>> # Config 3: (1,0,1)
        >>> prod3 = ProductOverGroupsWithOverlap(...)
        >>>
        >>> # Combine with sum
        >>> global_sum = GlobalSumOfProducts(
        ...     products=[prod1, prod2, prod3],
        ...     weights=[0.4, 0.35, 0.25],
        ...     device='cpu'
        ... )
        >>>
        >>> # Now cross-group dependencies can be detected!
        >>> x = torch.randn(100, 8)
        >>> log_p = global_sum.log_prob(x)

    Reference:
        Seng feedback (April 29, 2026): "sum nodes on top of the products
        that group these clusters"
    """

    def __init__(self, products, weights, device="cpu"):
        super().__init__()

        self.device = device
        self.num_products = len(products)

        # Store products as ModuleList for proper PyTorch registration
        self.products = nn.ModuleList(products)

        # Convert weights to tensor
        if isinstance(weights, np.ndarray):
            self.weights = torch.tensor(weights, dtype=torch.float32).to(device)
        else:
            self.weights = torch.tensor(list(weights), dtype=torch.float32).to(device)

        # Validation
        assert len(self.products) > 0, "Must have at least one product"
        assert len(self.products) == len(
            self.weights
        ), f"Mismatched products ({len(self.products)}) and weights ({len(self.weights)})"
        assert (
            abs(self.weights.sum().item() - 1.0) < 1e-5
        ), f"Weights must sum to 1, got {self.weights.sum().item()}"

        # Performance optimization: Pre-compute log weights (used in every forward pass)
        self.log_weights = torch.log(self.weights + 1e-9).unsqueeze(
            0
        )  # [1, num_products]

        logging.info(
            f"[GlobalSumOfProducts] Created with {self.num_products} products, "
            f"weights={self.weights.cpu().numpy()}"
        )

    def log_prob(self, x):
        """
        Compute log P(X) = log(Σ_c w_c × Product_c(X)).

        Algorithm:
            1. For each product c: compute log Product_c(x)
            2. Combine via logsumexp: log(Σ_c w_c × exp(log Product_c))

        Args:
            x (Tensor): [batch, d] input data

        Returns:
            log_prob (Tensor): [batch, 1] log probabilities
        """
        # Ensure tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Performance: Pre-allocate tensor instead of list append + cat
        ll_stack = torch.empty(
            x.shape[0], self.num_products, device=x.device, dtype=x.dtype
        )
        for i, product in enumerate(self.products):
            ll_stack[:, i : i + 1] = product.log_prob(x)  # [batch, 1]

        # LogSumExp: log(Σ_c w_c × exp(ll_c))
        # Performance: Use pre-computed log_weights from __init__
        log_prob = torch.logsumexp(ll_stack + self.log_weights, dim=1, keepdim=True)

        return log_prob

    def sample(self, n_samples):
        """
        Sample from the mixture by:
        1. Sampling which product to use (according to weights)
        2. Sampling from that product

        Args:
            n_samples (int): Number of samples to generate

        Returns:
            samples (Tensor): [n_samples, d] sampled data
        """
        with torch.no_grad():
            # Sample product indices according to mixture weights
            product_indices = torch.multinomial(
                self.weights, n_samples, replacement=True
            )

            # Sample from each product (batched for efficiency)
            samples = []
            for idx in range(self.num_products):
                n_from_this = (product_indices == idx).sum().item()
                if n_from_this > 0:
                    samples_from_product = self.products[idx].sample(n_from_this)
                    samples.append(samples_from_product)

            # Concatenate all samples
            all_samples = torch.cat(samples, dim=0)

            # Reshuffle to match original sampling order
            shuffle_back = torch.argsort(torch.argsort(product_indices))
            all_samples = all_samples[shuffle_back]

            return all_samples

    def get_size_bytes(self):
        """
        Estimate memory footprint (sum of all products).

        Returns:
            size_bytes (int): Total size in bytes
        """
        total_bytes = 0
        for product in self.products:
            if hasattr(product, "get_size_bytes"):
                total_bytes += product.get_size_bytes()
            else:
                buffer = io.BytesIO()
                torch.save(product.state_dict(), buffer)
                total_bytes += buffer.tell()
        return total_bytes


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

        # Convert to list for logging to avoid numpy/torch compatibility issues
        weights_list = self.weights.cpu().numpy().tolist()
        logging.info(f"[FedPC] EM Refined Weights: {weights_list}")

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

        # FIX: Add context column if components are hybrid mode classes
        # This ensures hybrid mode samples have shape [n, d+1] like horizontal mode
        # Check for ProductOverGroups/ProductOverGroupsWithOverlap (hybrid mode)
        # NOTE: FederatedProduct is used in VERTICAL mode and should NOT add context column
        is_hybrid = any(
            isinstance(c, (ProductOverGroups, ProductOverGroupsWithOverlap))
            for c in self.components
        )

        if is_hybrid:
            # Hybrid mode: Add context column with component indices
            # This makes shape consistent with horizontal [n, d+1]
            context_col = comp_indices.float().view(n, 1)
            samples = torch.cat([samples, context_col], dim=1)

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


def build_feature_indicator_matrix(X_splits, scenario, d_features=None):
    """
    Build indicator matrix M where M[k,j] = 1 if client k has feature j.

    Reference: Seng et al. (2025), Algorithm 1, Line 1 (implicit)

    Design Rationale:
        - Indicator matrix is the foundation for automatic feature grouping
        - Column j represents which clients have feature j
        - Used by group_features_by_client_set() to create feature subspaces

    Args:
        X_splits (List[np.ndarray]): Data splits per client
            - Horizontal: [n_k, d] - all have same features
            - Vertical: [n, d_k] - different features per client
            - Hybrid: varies
        scenario (str): 'horizontal', 'vertical', or 'hybrid'
        d_features (int): Total number of features (required for vertical/hybrid)

    Returns:
        M (np.ndarray): [K, d] binary indicator matrix
        feature_names (List[int]): Feature indices [0, 1, ..., d-1]

    Example (Vertical):
        >>> X_splits = [
        ...     np.random.randn(100, 3),  # Client 0: features [0,1,2]
        ...     np.random.randn(100, 2)   # Client 1: features [3,4]
        ... ]
        >>> M, names = build_feature_indicator_matrix(X_splits, 'vertical', d_features=5)
        >>> M
        array([[1, 1, 1, 0, 0],
               [0, 0, 0, 1, 1]])
    """
    K = len(X_splits)

    # Justification: Need total dimensionality to allocate M
    # For vertical/hybrid, must be provided explicitly
    if scenario == "horizontal":
        # Horizontal: all clients have same features
        d = X_splits[0].shape[1]
    else:
        # Vertical/Hybrid: must provide d_features
        if d_features is None:
            raise ValueError(
                f"d_features required for scenario='{scenario}'. "
                f"Provide total feature count."
            )
        d = d_features

    # Initialize indicator matrix
    # Justification: Binary matrix, start with all zeros
    M = np.zeros((K, d), dtype=int)

    if scenario == "horizontal":
        # Horizontal: All clients have all features
        # Justification: Same features across clients
        M[:, :] = 1  # All entries are 1

    elif scenario == "vertical":
        # Vertical: Auto-split features equally across clients
        # Justification: Standard vertical FL assumes equal split
        cols_per_client = np.array_split(range(d), K)
        for k in range(K):
            feature_indices = cols_per_client[k].tolist()
            M[k, feature_indices] = 1

    elif scenario == "hybrid":
        # Hybrid: Create overlapping feature splits
        # Justification: Hybrid FL has overlapping features across clients
        # Strategy: Each client gets base features + overlap with neighbors
        # This validates ProductOverGroupsWithOverlap architecture

        base_size = d // K
        overlap_size = max(1, d // (2 * K))  # ~15-20% overlap

        feature_sets = []
        for k in range(K):
            start = k * base_size
            end = min((k + 1) * base_size + overlap_size, d)
            features = list(range(start, end))
            feature_sets.append(features)

        # Ensure all features are covered
        all_features = set()
        for features in feature_sets:
            all_features.update(features)

        missing = set(range(d)) - all_features
        if missing:
            # Add missing features to last client
            feature_sets[-1].extend(sorted(missing))

        # Build indicator matrix
        for k in range(K):
            M[k, feature_sets[k]] = 1

        # Log overlap statistics
        total_refs = sum(len(fs) for fs in feature_sets)
        overlap_count = total_refs - d
        logging.info(
            f"Hybrid mode: Created overlapping feature splits with {overlap_count} overlaps"
        )
        for k, features in enumerate(feature_sets):
            logging.info(
                f"  Client {k}: features {features[:5]}...{features[-2:]} (size={len(features)})"
            )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    feature_names = list(range(d))
    return M, feature_names


def group_features_by_client_set(M, feature_names):
    """
    Group features by which clients have them (Algorithm 1 from Seng et al. 2025).

    Reference: Seng et al. (2025), Algorithm 1, Lines 3-6

    Algorithm:
        1. For each feature j: extract column M[:, j] (which clients have it)
        2. Group features with identical column patterns
        3. Convert column pattern to client set tuple
        4. Return mapping: client_set → [features]

    Mathematical Insight:
        Features with same column pattern M[:, j] share the same client set.
        These features should be modeled by a single GroupMixture over those clients.

    Design Rationale:
        - Automatic: No manual specification needed
        - Disjoint: Each feature appears in exactly one group (by construction)
        - Handles overlaps: If feature j appears in multiple clients,
          it gets grouped with other features having the same client set

    Args:
        M (np.ndarray): [K, d] indicator matrix
        feature_names (List[int]): Feature indices

    Returns:
        feature_subspaces (Dict): {
            (client_tuple): [feature_indices]
        }

    Example 1 (Vertical - Disjoint):
        >>> M = np.array([
        ...     [1, 1, 1, 0, 0],
        ...     [0, 0, 0, 1, 1]
        ... ])
        >>> group_features_by_client_set(M, list(range(5)))
        {
            (0,): [0, 1, 2],  # Features 0,1,2 only on client 0
            (1,): [3, 4]      # Features 3,4 only on client 1
        }

    Example 2 (Hybrid - Overlapping):
        >>> M = np.array([
        ...     [1, 1, 1, 0],
        ...     [0, 1, 1, 1]
        ... ])
        >>> group_features_by_client_set(M, list(range(4)))
        {
            (0,): [0],        # Feature 0 only on client 0
            (0, 1): [1, 2],   # Features 1,2 on BOTH clients (overlap!)
            (1,): [3]         # Feature 3 only on client 1
        }

        Result: GroupMixture for (0,1) combines both clients for features [1,2]
    """
    K, d = M.shape

    # Step 1: Find distinct column patterns
    # Justification: Features with same column pattern share same client set
    feature_to_clients = {}
    for j in range(d):
        # Extract column j (which clients have feature j)
        col = tuple(M[:, j])  # e.g., (1, 0, 1) means clients 0 and 2 have it

        if col not in feature_to_clients:
            feature_to_clients[col] = []
        feature_to_clients[col].append(feature_names[j])

    # Step 2: Convert column patterns to client sets
    # Justification: Client set is more interpretable than binary pattern
    feature_subspaces = {}
    for col_pattern, features in feature_to_clients.items():
        # col_pattern is (1, 0, 1, ...) → client_set is (0, 2, ...)
        # Justification: Extract indices where col_pattern[k] == 1
        client_set = tuple(k for k in range(K) if col_pattern[k] == 1)

        # Skip invalid patterns (no clients have these features)
        # Justification: Features must belong to at least one client
        if len(client_set) == 0:
            logging.warning(f"Features {features} have no clients! Skipping.")
            continue

        feature_subspaces[client_set] = features

    # Log results for debugging
    logging.info(
        f"[FedPC] Automatic feature grouping: {len(feature_subspaces)} subspaces"
    )
    for clients, features in feature_subspaces.items():
        logging.info(f"  Clients {clients} share features {features}")

    return feature_subspaces


def sample_cluster_combinations(K_clients, K_local, num_samples=10, seed=42):
    """
    Sample cluster combinations for sum-over-products in hybrid mode.

    Following Seng's guidance: "combine these clusters 'randomly' (since pairing
    each cluster from client i with each cluster from client j is too demanding)"

    Strategy:
        - If K_local^K_clients ≤ 20: Enumerate all combinations (exact)
        - Otherwise: Randomly sample num_samples combinations (approximate)

    Mathematical Context:
        Each combination represents a different "factorization" of the data:
        - Combination (0,0,0): All clients use cluster 0
        - Combination (0,1,1): Client 0 uses cluster 0, others use cluster 1
        - etc.

        The sum over these combinations breaks independence between feature groups!

    Args:
        K_clients (int): Number of clients
        K_local (int): Number of local clusters per client
        num_samples (int): How many combinations to sample if not enumerating all
        seed (int): Random seed for reproducibility

    Returns:
        combinations (List[Tuple[int]]): List of cluster configurations
            Each tuple has K_clients elements, specifying which cluster
            each client contributes to this product
        weights (np.ndarray): Uniform weights for each combination (sum to 1)

    Example:
        >>> # For K=3 clients, K_local=2 clusters per client
        >>> combos, weights = sample_cluster_combinations(3, 2)
        >>> # Output: 8 combinations (enumerate all since 2^3 = 8 ≤ 20)
        >>> combos
        [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]
        >>> weights
        array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])

    Reference:
        Seng feedback (April 29, 2026): "combine these clusters 'randomly'"
    """
    np.random.seed(seed)

    max_combinations = K_local**K_clients

    # Decision: Enumerate if small enough, sample if too large
    if max_combinations <= 20:
        # Enumerate all combinations
        import itertools

        combinations = list(itertools.product(range(K_local), repeat=K_clients))
        logging.info(
            f"[ClusterCombinations] Enumerating all {len(combinations)} combinations "
            f"(K_clients={K_clients}, K_local={K_local})"
        )
    else:
        # Random sampling without replacement
        num_samples = min(num_samples, max_combinations)
        combinations = set()

        # Sample unique combinations
        while len(combinations) < num_samples:
            config = tuple(np.random.randint(0, K_local) for _ in range(K_clients))
            combinations.add(config)

        combinations = list(combinations)
        logging.info(
            f"[ClusterCombinations] Sampled {len(combinations)}/{max_combinations} combinations "
            f"(K_clients={K_clients}, K_local={K_local})"
        )

    # Uniform weights (all combinations equally likely)
    # Justification: Seng mentions "randomly", no principled weighting yet
    weights = np.ones(len(combinations)) / len(combinations)

    return combinations, weights


def compute_adaptive_hyperparameters(
    mode: str,
    num_features: int,
    num_samples: int,
    data_type: str,
    base_num_sums: int = 20,
    base_num_leaves: int = 20,
    base_epochs: int = 100,
    base_depth: int = None,
) -> Dict[str, Any]:
    """
    Compute adaptive hyperparameters following 5-criterion system for FedCDH SPNs.

    This implements the documented adaptive capacity scaling system that addresses
    horizontal mode underperformance (F1=0.133-0.255 → target 0.5+).

    The 5 criteria are:
    1. Mode-Specific Base Capacity: Different architectures for horizontal/vertical/hybrid
    2. Sample-to-Feature Ratio Scaling: Adjust capacity based on data availability
    3. Data Type Differentiation: Linear vs nonlinear complexity adjustments
    4. Quality-Aware Epoch Scheduling: Mode and feature-dependent training duration
    5. Mode-Aware Regularization: Dropout and weight decay tuned per scenario

    Args:
        mode: Federated scenario mode ("horizontal", "vertical", "hybrid")
        num_features: Number of features in local data (d_k)
        num_samples: Number of samples in local data (n_k)
        data_type: Data generation type ("linear" or "nonlinear")
        base_num_sums: Base number of sum nodes (default 20)
        base_num_leaves: Base number of leaf nodes (default 20)
        base_epochs: Base training epochs (default 100)
        base_depth: Base tree depth (default None, auto-computed from features)

    Returns:
        Dictionary with keys:
            - num_sums: Adapted number of sum nodes
            - num_leaves: Adapted number of leaf nodes
            - depth: Adapted tree depth
            - epochs: Adapted training epochs
            - dropout: Dropout rate for regularization
            - weight_decay: L2 regularization weight

    Example:
        >>> params = compute_adaptive_hyperparameters(
        ...     mode="horizontal",
        ...     num_features=10,
        ...     num_samples=400,
        ...     data_type="linear"
        ... )
        >>> print(params['num_sums'])  # Expected: 40+ (4×10)
        >>> print(params['dropout'])   # Expected: 0.1 (ratio 40 < 100)

    References:
        - agents/working_state.md: "5-Criterion Adaptive Hyperparameter System"
        - v2 experiment proposal: Horizontal mode performance fix
    """

    # Criterion 1: Mode-Specific Base Capacity
    # Horizontal: Many features need broader representation (4×d)
    # Vertical: Few features need depth (8×d for d>3, otherwise small)
    # Hybrid: Intermediate complexity (6×d)
    if mode == "horizontal":
        base_sums = max(32, 4 * num_features)
        base_leaves = max(16, 2 * num_features)
    elif mode == "vertical":
        if num_features <= 3:
            base_sums = 8
            base_leaves = 8
        else:
            base_sums = 8 * num_features
            base_leaves = 4 * num_features
    else:  # hybrid
        base_sums = 6 * num_features
        base_leaves = 3 * num_features

    # Criterion 2: Sample-to-Feature Ratio Scaling
    # Low ratio (< 50): Risk of overfitting, reduce capacity
    # Medium ratio (50-200): Standard capacity
    # High ratio (> 200): Can afford more capacity for complex patterns
    ratio = num_samples / max(1, num_features)
    if ratio < 50:
        capacity_scale = 0.5
    elif ratio < 100:
        capacity_scale = 0.75
    elif ratio < 200:
        capacity_scale = 1.0
    else:
        capacity_scale = min(1.5, 1.0 + (ratio - 200) / 400)

    final_num_sums = int(base_sums * capacity_scale)
    final_num_leaves = int(base_leaves * capacity_scale)

    # Criterion 3: Data Type Differentiation
    # Nonlinear data needs deeper trees and more training
    # Linear data can use shallower structures
    if base_depth is None:
        base_depth = int(np.floor(np.log2(max(1, num_features))))

    if data_type == "nonlinear":
        depth_bonus = 1
        epoch_multiplier = 1.3
    else:  # linear
        depth_bonus = 0
        epoch_multiplier = 1.0

    final_depth = max(1, base_depth + depth_bonus)

    # Criterion 4: Quality-Aware Epoch Scheduling
    # More features need more training, especially in horizontal mode
    # Base scaling: d^1.5 (superlinear growth)
    epoch_base_scale = (num_features / 5.0) ** 1.5

    # Mode-specific multipliers
    if mode == "horizontal":
        # Horizontal needs more epochs due to broader feature space
        mode_multiplier = 1.0 + num_features / 30
    elif mode == "vertical":
        # Vertical can train faster (fewer features)
        mode_multiplier = 0.8
    else:  # hybrid
        mode_multiplier = 0.9

    final_epochs = int(
        base_epochs * epoch_base_scale * epoch_multiplier * mode_multiplier
    )
    final_epochs = max(100, min(final_epochs, 500))  # Clamp to [100, 500]

    # Criterion 5: Mode-Aware Regularization
    # Horizontal: Moderate regularization (many features, risk of spurious correlations)
    # Vertical: Higher regularization (few features, risk of overfitting to noise)
    # Hybrid: Light regularization (balanced scenario)
    if mode == "horizontal":
        weight_decay = 1e-4
        dropout = 0.1 if ratio < 100 else 0.0
    elif mode == "vertical":
        weight_decay = 1e-3  # Higher for vertical (fewer features)
        dropout = 0.0  # Vertical typically has enough regularization from structure
    else:  # hybrid
        weight_decay = 5e-5
        dropout = 0.05 if ratio < 100 else 0.0

    return {
        "num_sums": final_num_sums,
        "num_leaves": final_num_leaves,
        "depth": final_depth,
        "epochs": final_epochs,
        "dropout": dropout,
        "weight_decay": weight_decay,
    }


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
                    ll = spn.log_prob(x_obs)
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
