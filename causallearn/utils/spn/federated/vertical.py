"""
Vertical Federated SPN implementations.

This module implements the vertical federated learning scenario where
clients have different features but the same samples.
"""

import logging
import numpy as np
from typing import List, Dict
import torch
import torch.nn as nn
from ..core.local import LocalSPNWrapper, LocalClusterMixture


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
        self,
        client_spns,
        weights,
        feature_indices,
        device="cpu",
        full_d=None,
        use_nan_masking=None,
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
        self.full_d = full_d  # Total number of features (for context stripping)

        # Determine whether to use NaN masking or feature extraction
        # If not explicitly specified, default to False (feature extraction)
        # Hybrid mode must explicitly pass use_nan_masking=True
        self.use_nan_masking = use_nan_masking if use_nan_masking is not None else False

        # Performance optimization: Pre-compute log weights (called in every forward pass)
        self.log_weights = torch.log(self.weights + 1e-9).unsqueeze(0)  # [1, K]

        # Performance optimization: Pre-compute feature mask for NaN tensor creation
        # This avoids creating the mask on every forward pass
        if self.use_nan_masking and self.full_d is not None:
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
        # Input may be [batch, d+1] where last column is context. Strip it before processing.
        if self.full_d is not None and x.shape[1] > self.full_d:
            # Context column detected - strip it
            x = x[:, : self.full_d]

        # Step 1: Extract features for this group
        # Justification: Each group only models its subset of features
        # Use NaN masking if requested (hybrid mode), otherwise extract features (vertical mode)
        if self.use_nan_masking and self.full_d is not None:
            # Hybrid mode: SPNs trained on full features, use NaN masking for marginalization
            # Performance optimization: Clone and mask instead of full + copy
            x_g = x.clone()  # [batch, d]
            x_g[:, self.nan_mask] = float("nan")  # Mask features not in this group
        else:
            # Vertical mode: SPNs trained on feature subspace, extract features
            # Note: During CI testing, we receive full-dimensional data [batch, d_full]
            # and extract the subset for this client. This is expected behavior.

            # Validate: Input must have enough dimensions for feature extraction
            if x.shape[1] <= max(self.feature_indices):
                import logging

                logging.error(
                    f"REAL DIMENSION ERROR: Cannot extract features {self.feature_indices} from input shape {x.shape}. "
                    f"Need at least {max(self.feature_indices)+1} features but got {x.shape[1]}."
                )
                raise IndexError(
                    f"Cannot extract features {self.feature_indices} from tensor with shape {x.shape}"
                )

            # Debug log: Track when we extract features from full-dimensional input
            if x.shape[1] != len(self.feature_indices):
                import logging

                logging.debug(
                    f"Vertical mode: extracting features {self.feature_indices} from full tensor with shape {x.shape}"
                )

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
            # Check if client SPN expects different dimensions
            # This happens in hybrid mode where client has more features than this group
            spn_input = x_g

            # Get expected dimensions from SPN
            if isinstance(spn, LocalClusterMixture):
                # LocalClusterMixture wraps LocalSPNWrapper - check first cluster SPN
                if len(spn.cluster_spns) > 0:
                    first_spn = spn.cluster_spns[0]
                    if hasattr(first_spn, "mean") and first_spn.mean is not None:
                        expected_dims = first_spn.mean.shape[0]
                    else:
                        expected_dims = x_g.shape[1]  # Default to x_g dims
                else:
                    expected_dims = x_g.shape[1]
            elif hasattr(spn, "mean") and spn.mean is not None:
                expected_dims = spn.mean.shape[0]
            else:
                expected_dims = x_g.shape[1]  # Default to x_g dims

            if expected_dims > x_g.shape[1]:
                # Pad with NaN to marginalize over extra features
                padding_size = expected_dims - x_g.shape[1]
                padding = torch.full(
                    (x_g.shape[0], padding_size),
                    float("nan"),
                    device=x_g.device,
                    dtype=x_g.dtype,
                )
                spn_input = torch.cat([x_g, padding], dim=1)

            ll_stack[:, i : i + 1] = spn.log_prob(spn_input)  # [batch, 1]

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

            # Extract relevant features from SPN samples
            d_sampled = comp_samples.shape[1]
            d_expected = len(self.feature_indices)

            if d_sampled == d_expected:
                # Dimensions match - samples are already correct
                pass
            elif self.full_d is not None and d_sampled == self.full_d:
                # SPN returned full-dimensional samples (vertical mode)
                # Extract using global feature indices
                comp_samples = comp_samples[:, self.feature_indices]
            elif d_sampled > d_expected:
                # SPN returned more features than needed (hybrid mode)
                # Extract first d_expected features (assumes features are ordered)
                comp_samples = comp_samples[:, :d_expected]
            else:
                # SPN returned fewer features than expected
                # Pad with zeros (shouldn't normally happen)
                padding = torch.zeros(
                    comp_samples.shape[0],
                    d_expected - d_sampled,
                    device=comp_samples.device,
                )
                comp_samples = torch.cat([comp_samples, padding], dim=1)

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
