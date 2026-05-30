"""
Cluster-Conditional Federated SPN (Gap 4).

This module implements cluster-conditional vertical federated SPNs.
"""

import logging
import numpy as np
from typing import List, Dict
import torch
import torch.nn as nn
from ..core.local import LocalClusterMixture


class FederatedProductWithClusters(nn.Module):
    """
    Cluster-Conditional Product-of-Experts for Vertical FL (Seng et al. Assumption 2).

    Addresses Gap 4: Enables capturing cross-client dependencies in vertical mode
    through cluster-conditional factorization.

    Factorization:
        p(X₁,...,Xₙ) = Σₗ q(L=l) Πⁿᵢ₌₁ p(Xᵢ|L=l)

    Where:
    - L: Latent cluster variable (mechanism)
    - Xᵢ: Features owned by client i
    - q(L): Mixture weights over clusters
    - p(Xᵢ|L): Client i's local SPN conditioned on cluster

    This allows capturing dependencies across clients while maintaining:
    1. Tractability (product structure)
    2. Privacy (no raw data sharing)
    3. Expressivity (cluster-specific dependencies)

    Reference: Seng et al. (2025), Assumption 2
    """

    def __init__(
        self,
        local_cluster_models: List,  # List of LocalClusterMixture per client
        feature_maps: Dict[int, List[int]],
        num_features: int,
        num_clusters: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.local_cluster_models = nn.ModuleList(local_cluster_models)
        self.feature_maps = feature_maps
        self.num_features = num_features
        self.num_clusters = num_clusters
        self.device = device

        # Learnable cluster mixture weights q(L)
        # Initialized from local sample counts
        self.cluster_logits = nn.Parameter(torch.zeros(num_clusters, device=device))

        # Initialize from local cluster sizes
        self._initialize_cluster_weights(local_cluster_models)

        self.to(device)

    def _initialize_cluster_weights(self, local_models):
        """Initialize q(L) from aggregated local cluster sample counts."""
        cluster_counts = torch.zeros(self.num_clusters, device=self.device)

        for local_model in local_models:
            if hasattr(local_model, "cluster_sample_counts"):
                # Aggregate sample counts across clients
                counts = torch.tensor(
                    local_model.cluster_sample_counts, device=self.device
                )
                cluster_counts += counts

        # Convert to logits (unnormalized log probabilities)
        if cluster_counts.sum() > 0:
            cluster_probs = cluster_counts / cluster_counts.sum()
            # Add small epsilon to avoid log(0)
            self.cluster_logits.data = torch.log(cluster_probs + 1e-9)

    def log_prob(self, x):
        """
        Compute log p(X) using cluster-conditional factorization.

        log p(X) = log Σₗ q(L=l) Πᵢ p(Xᵢ|L=l)
                 = log Σₗ [q(L=l) × exp(Σᵢ log p(Xᵢ|L=l))]
                 = logsumexp_l [log q(L=l) + Σᵢ log p(Xᵢ|L=l)]

        Args:
            x: Input tensor [batch_size, num_features] or with NaN masking

        Returns:
            log_prob: Tensor [batch_size, 1]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)

        batch_size = x.shape[0]

        # Compute log q(L) for all clusters
        log_q_L = torch.log_softmax(self.cluster_logits, dim=0)  # [num_clusters]

        # Compute log p(Xᵢ|L=l) for each client and cluster
        # Shape: [num_clients, num_clusters, batch_size]
        client_cluster_lls = []

        for client_id, local_model in enumerate(self.local_cluster_models):
            # Extract client's features
            client_features = self.feature_maps[client_id]
            x_client = x[:, client_features]  # [batch_size, d_client]

            # Get log p(Xᵢ|L=l) for all clusters
            # LocalClusterMixture has K cluster-specific SPNs
            cluster_lls = local_model.get_cluster_log_probs(
                x_client
            )  # [num_clusters, batch_size]
            client_cluster_lls.append(cluster_lls)

        # Stack: [num_clients, num_clusters, batch_size]
        client_cluster_lls = torch.stack(client_cluster_lls, dim=0)

        # Sum over clients (product in log space): Πᵢ p(Xᵢ|L) = exp(Σᵢ log p(Xᵢ|L))
        # Shape: [num_clusters, batch_size]
        sum_log_probs = client_cluster_lls.sum(dim=0)

        # Add cluster prior: log q(L) + Σᵢ log p(Xᵢ|L)
        # Shape: [num_clusters, batch_size]
        log_joint = log_q_L.unsqueeze(1) + sum_log_probs

        # Marginalize over clusters: log Σₗ exp(...)
        # Shape: [batch_size]
        log_prob = torch.logsumexp(log_joint, dim=0)

        return log_prob.unsqueeze(1)  # [batch_size, 1]

    def sample(self, n):
        """
        Sample from cluster-conditional model:
        1. Sample cluster l ~ q(L)
        2. For each client i, sample Xᵢ ~ p(Xᵢ|L=l)
        3. Concatenate client samples into full feature vector
        """
        # Sample cluster assignments
        cluster_probs = torch.softmax(self.cluster_logits, dim=0).cpu().numpy()
        cluster_assignments = np.random.choice(
            self.num_clusters, size=n, p=cluster_probs
        )

        # Sample from each client conditioned on cluster
        samples = torch.zeros(n, self.num_features, device=self.device)

        for client_id, local_model in enumerate(self.local_cluster_models):
            client_features = self.feature_maps[client_id]

            # Sample from cluster-specific SPNs
            client_samples = local_model.sample_from_clusters(
                cluster_assignments
            )  # [n, d_client]

            samples[:, client_features] = client_samples

        return samples
