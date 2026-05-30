"""
Gap 2 Fix: Latent Routing Layer for Cross-Silo Confounders.

This module implements server-side hidden latent variables to capture
dependencies between features split across vertical FL clients.

Problem:
    Standard product nodes assume P(X,Y) = P(X)P(Y) for vertically split
    features, missing confounders and yielding catastrophic SHD on benchmarks
    like Sachs and DreamNet.

Solution:
    Introduce latent routing: P(X,Y) = Σ_l P(L=l) P(X|L=l) P(Y|L=l)
    Latent L acts as bottleneck capturing cross-client dependencies.

Reference: Gap Analysis - Cross-Silo Unobserved Confounder Trap
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict


class LatentFederatedProductNode(nn.Module):
    """
    Product node with latent routing for vertical FL.

    Mathematical Form:
        P(X₁, X₂, ..., Xₙ) = Σₗ P(L=l) ∏ᵢ P(Xᵢ | L=l)

    Where:
        - L: Discrete latent variable (cluster/mechanism)
        - Xᵢ: Features owned by client i
        - P(L): Learnable mixture weights
        - P(Xᵢ|L): Client i's cluster-conditional SPN

    Key Difference from Standard Product:
        - Standard: P(X,Y) = P(X) × P(Y)  [assumes independence]
        - Latent:   P(X,Y) = Σₗ P(L) × P(X|L) × P(Y|L)  [allows dependencies via L]

    This captures confounding without data exchange:
        - L routes information between silos
        - Clients never see each other's data
        - Dependencies learned via EM on aggregated statistics
    """

    def __init__(
        self,
        local_cluster_models: List[nn.Module],
        feature_maps: Dict[int, List[int]],
        num_features: int,
        num_latent_states: int = 4,
        device: str = "cpu",
    ):
        """
        Args:
            local_cluster_models: List of LocalClusterMixture (one per client)
                Each has K cluster-specific SPNs: P(Xᵢ | L=k)
            feature_maps: {client_id: [feature_indices]}
            num_features: Total features across all clients
            num_latent_states: Number of latent mechanisms |L|
            device: CPU or CUDA
        """
        super().__init__()

        self.local_cluster_models = nn.ModuleList(local_cluster_models)
        self.feature_maps = feature_maps
        self.num_features = num_features
        self.num_latent_states = num_latent_states
        self.device = device

        # Learnable latent mixture weights P(L)
        # Initialized uniformly, refined via EM
        self.latent_logits = nn.Parameter(torch.zeros(num_latent_states, device=device))

        # Initialize from local cluster assignments if available
        self._initialize_latent_weights()

        self.to(device)

    def _initialize_latent_weights(self):
        """
        Initialize P(L) from aggregated local cluster sample counts.

        If local models have cluster statistics, use them.
        Otherwise, start with uniform distribution.
        """
        cluster_counts = torch.zeros(self.num_latent_states, device=self.device)

        for local_model in self.local_cluster_models:
            if hasattr(local_model, "cluster_sample_counts"):
                counts = torch.tensor(
                    local_model.cluster_sample_counts,
                    dtype=torch.float32,
                    device=self.device,
                )
                # Add to global count
                if len(counts) == self.num_latent_states:
                    cluster_counts += counts

        # Convert to logits
        if cluster_counts.sum() > 0:
            probs = cluster_counts / cluster_counts.sum()
            self.latent_logits.data = torch.log(probs + 1e-9)
            logging.info(
                f"[LatentRouting] Initialized P(L) from cluster counts: "
                f"{probs.cpu().numpy()}"
            )
        else:
            # Uniform initialization
            logging.info(
                f"[LatentRouting] Initialized P(L) uniformly over "
                f"{self.num_latent_states} states"
            )

    def log_prob(self, x):
        """
        Compute log P(X₁, ..., Xₙ) with latent routing.

        Algorithm:
            1. For each latent state l ∈ L:
                a. Compute P(Xᵢ | L=l) for each client i
                b. Product: P(X₁,...,Xₙ | L=l) = ∏ᵢ P(Xᵢ | L=l)
            2. Marginalize: P(X) = Σₗ P(L=l) × P(X | L=l)

        Mathematical Detail:
            log P(X) = log Σₗ [P(L=l) × ∏ᵢ P(Xᵢ|L=l)]
                     = log Σₗ [P(L=l) × exp(Σᵢ log P(Xᵢ|L=l))]
                     = logsumexp_l [log P(L=l) + Σᵢ log P(Xᵢ|L=l)]

        Args:
            x: Full feature tensor [batch_size, num_features]

        Returns:
            log_prob: [batch_size, 1]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        batch_size = x.shape[0]

        # Compute log P(L) for all latent states
        log_p_L = torch.log_softmax(self.latent_logits, dim=0)  # [num_latent_states]

        # Compute log P(Xᵢ | L=l) for each client and latent state
        # Shape: [num_clients, num_latent_states, batch_size]
        client_latent_lls = []

        for client_id, local_model in enumerate(self.local_cluster_models):
            # Extract client's features
            client_features = self.feature_maps[client_id]
            x_client = x[:, client_features]  # [batch_size, d_client]

            # Get log P(Xᵢ | L=l) for all latent states
            # LocalClusterMixture has K cluster-specific SPNs
            latent_lls = local_model.get_cluster_log_probs(
                x_client
            )  # [num_latent_states, batch_size]

            client_latent_lls.append(latent_lls)

        # Stack: [num_clients, num_latent_states, batch_size]
        client_latent_lls = torch.stack(client_latent_lls, dim=0)

        # Product over clients (sum in log space): ∏ᵢ P(Xᵢ|L) = exp(Σᵢ log P(Xᵢ|L))
        # Shape: [num_latent_states, batch_size]
        sum_log_probs = client_latent_lls.sum(dim=0)

        # Add latent prior: log P(L) + Σᵢ log P(Xᵢ|L)
        # Shape: [num_latent_states, batch_size]
        log_joint = log_p_L.unsqueeze(1) + sum_log_probs

        # Marginalize over latent states: log Σₗ exp(...)
        # Shape: [batch_size]
        log_prob = torch.logsumexp(log_joint, dim=0)

        return log_prob.unsqueeze(1)  # [batch_size, 1]

    def forward_cross_silo(
        self,
        client_a_marginal: torch.Tensor,
        client_b_marginal: torch.Tensor,
        latent_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified cross-silo computation for two clients.

        This is a convenience method for analyzing pairwise dependencies.

        Formula:
            P(Xₐ, Xᵦ) = Σₗ P(L=l) × P(Xₐ|L=l) × P(Xᵦ|L=l)

        Args:
            client_a_marginal: [batch, num_latent_states] - P(Xₐ|L)
            client_b_marginal: [batch, num_latent_states] - P(Xᵦ|L)
            latent_weights: [num_latent_states] - P(L)

        Returns:
            joint_density: [batch] - P(Xₐ, Xᵦ)
        """
        # Compute joint conditional: P(Xₐ, Xᵦ | L) = P(Xₐ|L) × P(Xᵦ|L)
        # Shape: [batch, num_latent_states]
        joint_conditional = client_a_marginal * client_b_marginal

        # Weight by latent prior and marginalize
        # Shape: [batch]
        global_joint_density = torch.sum(
            joint_conditional * latent_weights.unsqueeze(0), dim=1
        )

        return global_joint_density

    def sample(self, n: int) -> torch.Tensor:
        """
        Sample from latent-routed distribution.

        Algorithm:
            1. Sample latent assignments: l ~ P(L)
            2. For each client i, sample Xᵢ ~ P(Xᵢ | L=l)
            3. Concatenate client samples

        Args:
            n: Number of samples

        Returns:
            samples: [n, num_features]
        """
        # Sample latent assignments
        latent_probs = torch.softmax(self.latent_logits, dim=0).cpu().numpy()
        latent_assignments = np.random.choice(
            self.num_latent_states, size=n, p=latent_probs
        )

        # Initialize sample tensor
        samples = torch.zeros(n, self.num_features, device=self.device)

        # Sample from each client conditioned on latent
        for client_id, local_model in enumerate(self.local_cluster_models):
            client_features = self.feature_maps[client_id]

            # Sample from cluster-conditional model
            client_samples = []
            for i, l in enumerate(latent_assignments):
                # Get cluster-specific SPN for latent state l
                cluster_spn = local_model.cluster_spns[l]
                sample = cluster_spn.sample(1)  # [1, d_client]
                client_samples.append(sample)

            client_samples = torch.cat(client_samples, dim=0)  # [n, d_client]

            # Place in global sample tensor
            samples[:, client_features] = client_samples

        return samples

    def train_latent_weights_em(self, x: torch.Tensor, epochs: int = 10):
        """
        Refine latent weights P(L) via EM.

        E-Step: Compute posterior P(L | X)
        M-Step: Update P(L) = E[P(L | X)]

        Args:
            x: Full dataset [n, d]
            epochs: EM iterations
        """
        with torch.no_grad():
            for epoch in range(epochs):
                # E-Step: Compute responsibilities
                log_p_L = torch.log_softmax(self.latent_logits, dim=0)

                # Compute log P(X, L)
                client_latent_lls = []
                for client_id, local_model in enumerate(self.local_cluster_models):
                    client_features = self.feature_maps[client_id]
                    x_client = x[:, client_features]
                    latent_lls = local_model.get_cluster_log_probs(x_client)
                    client_latent_lls.append(latent_lls)

                client_latent_lls = torch.stack(client_latent_lls, dim=0)
                sum_log_probs = client_latent_lls.sum(dim=0)  # [L, N]
                log_joint = log_p_L.unsqueeze(1) + sum_log_probs  # [L, N]

                # Compute posterior: P(L | X)
                log_evidence = torch.logsumexp(log_joint, dim=0, keepdim=True)  # [1, N]
                log_posterior = log_joint - log_evidence  # [L, N]
                responsibilities = torch.exp(log_posterior)  # [L, N]

                # M-Step: Update P(L)
                new_latent_probs = responsibilities.mean(dim=1)  # [L]
                self.latent_logits.data = torch.log(new_latent_probs + 1e-9)

        logging.info(
            f"[LatentRouting] EM refined P(L): "
            f"{torch.softmax(self.latent_logits, dim=0).cpu().numpy()}"
        )


def replace_product_with_latent_routing(
    federated_product, num_latent_states: int = 4, train_data: torch.Tensor = None
) -> LatentFederatedProductNode:
    """
    Convert standard FederatedProduct to latent-routed version.

    Args:
        federated_product: Existing FederatedProduct instance
        num_latent_states: Number of latent mechanisms
        train_data: Optional data for EM initialization

    Returns:
        latent_product: LatentFederatedProductNode with same clients
    """
    # Extract components from existing product
    local_models = list(federated_product.local_models)
    feature_maps = federated_product.feature_maps
    num_features = federated_product.num_features
    device = federated_product.device

    # Create latent-routed version
    latent_product = LatentFederatedProductNode(
        local_cluster_models=local_models,
        feature_maps=feature_maps,
        num_features=num_features,
        num_latent_states=num_latent_states,
        device=device,
    )

    # Optionally refine with EM
    if train_data is not None:
        logging.info("[LatentRouting] Refining latent weights with EM...")
        latent_product.train_latent_weights_em(train_data, epochs=10)

    return latent_product
