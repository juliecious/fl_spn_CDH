"""
Hybrid Federated SPN implementations.

This module implements the hybrid federated learning scenario with
both horizontal and vertical partitioning.
"""

import logging
import numpy as np
from typing import List, Dict, Optional
import torch
import torch.nn as nn
from ..core.local import LocalClusterMixture


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
