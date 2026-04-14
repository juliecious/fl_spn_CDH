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

    def __init__(self, client_spns, weights, feature_indices, device="cpu"):
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

        Algorithm:
            1. Extract features: x_g = x[:, feature_indices]
            2. For each client k: compute log P_k(x_g)
            3. Compute log Σ_k [w_k × P_k(x_g)] via logsumexp trick

        Mathematical Detail:
            log Σ_k [w_k × P_k] = logsumexp_k(log w_k + log P_k)

        Args:
            x (Tensor): [batch, d_full] full feature matrix

        Returns:
            log_prob (Tensor): [batch, 1] log probabilities for this group

        Reference: Seng et al. (2025), Definition 1 (Horizontal FL mixture)
        """
        # Step 1: Extract features for this group
        # Justification: Each group only models its subset of features
        x_g = x[:, self.feature_indices]  # [batch, len(feature_indices)]

        # Step 2: Compute log-likelihoods from each client SPN
        # Justification: Each client contributes its learned distribution
        client_lls = []
        for spn in self.client_spns:
            ll = spn.log_prob(x_g)  # [batch, 1]
            client_lls.append(ll)

        # Step 3: Stack and compute weighted mixture via logsumexp
        # Justification: Numerically stable computation of log(Σ exp(...))
        ll_stack = torch.cat(client_lls, dim=1)  # [batch, K]
        log_weights = torch.log(self.weights + 1e-9).unsqueeze(0)  # [1, K]

        # log Σ_k [w_k × P_k] = logsumexp(log w_k + log P_k)
        log_prob = torch.logsumexp(ll_stack + log_weights, dim=1, keepdim=True)

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
            comp_samples = spn.sample(count.item())  # [count, len(feature_indices)]

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
            group_samples = mixture_g.sample(n)  # [n, d_g]

            # Place samples in correct feature positions
            # Justification: Each group models specific features (stored in feature_groups)
            indices = self.feature_groups[g]
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
        # Check for both old (FederatedProduct) and new (ProductOverGroups, ProductOverGroupsWithOverlap) classes
        is_hybrid = any(
            isinstance(
                c, (FederatedProduct, ProductOverGroups, ProductOverGroupsWithOverlap)
            )
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
        # Hybrid: Infer from data shapes
        # Justification: Each client's data reveals which features they have
        # Assumption: Features are ordered, client k gets indices proportional to position
        # This is a simplification; in practice, feature mapping would be provided

        # For now, use equal split (conservative approach)
        # Real implementation would receive explicit feature mapping
        cols_per_client = np.array_split(range(d), K)
        for k in range(K):
            M[k, cols_per_client[k].tolist()] = 1

        logging.info(
            "Hybrid mode: Using equal feature split for indicator matrix. "
            "For overlapping features, provide explicit feature_maps."
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
