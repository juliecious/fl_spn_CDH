"""
Gap 1 Fix: Structural Synchronization for Horizontal Aggregation.

This module implements deterministic structure templates to prevent
parameter averaging across misaligned SPN architectures.

Problem:
    In horizontal FL with non-IID clients, averaging parameters across
    structurally different SPNs degenerates P(X,Y) estimation, breaking
    causal discovery via excessive false positives.

Solution:
    Broadcast unified RAT-SPN template with deterministic seed.
    Reserve root as latent mixture over client invariant embeddings.

Reference: Gap Analysis - Structural Incoherence in Horizontal Aggregation
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple
from simple_einet.einet import EinetConfig


class FederatedPCStructureManager:
    """
    Manages structural alignment across federated clients.

    Core Principle:
        All horizontal clients must instantiate IDENTICAL SPN structures
        to enable valid parameter aggregation and tractable joint inference.

    Approach:
        1. Server generates deterministic region graph template
        2. Broadcast seed + config to all clients
        3. Clients instantiate SPNs with shared structure
        4. Server aggregates via convex combination (data-weighted)
    """

    def __init__(self, variable_num: int, random_seed: int = 42):
        """
        Args:
            variable_num: Number of variables (features) in dataset
            random_seed: Deterministic seed for structural alignment
        """
        self.variable_num = variable_num
        self.seed = random_seed

    def generate_shared_region_graph(
        self,
        depth: int = 2,
        num_sums: int = 20,
        num_leaves: int = 20,
        num_repetitions: int = 10,
    ) -> Dict:
        """
        Generate unified structural template for all clients.

        Returns deterministic region splits and scope assignments matching
        standard RAT-SPN / EiNet architecture patterns.

        Args:
            depth: Tree depth
            num_sums: Sum nodes per layer
            num_leaves: Leaf distributions per region
            num_repetitions: Repetitions for ensemble

        Returns:
            structure_template: {
                'seed': Random seed for structure,
                'scopes': List of variable indices,
                'split_points': Deterministic variable ordering,
                'config': EiNet configuration dict,
            }
        """
        # Set deterministic seed for structural generation
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Construct scope assignments
        scopes = list(range(self.variable_num))

        # Generate deterministic variable ordering for region splits
        # This ensures all clients partition feature space identically
        split_points = np.random.permutation(scopes).tolist()

        # Create EiNet config for structural consistency
        config = EinetConfig(
            num_features=self.variable_num,
            num_channels=1,
            num_sums=num_sums,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
            depth=depth,
            num_classes=1,
            leaf_type="Normal",  # String identifier for serialization
            layer_type="linsum",
            structure="poon-domingos",  # Deterministic structure
        )

        structure_template = {
            "seed": self.seed,
            "scopes": scopes,
            "split_points": split_points,
            "config": config.__dict__,  # Serialize for transmission
            "num_features": self.variable_num,
        }

        logging.info(
            f"[StructureSync] Generated shared template: "
            f"seed={self.seed}, features={self.variable_num}, depth={depth}"
        )

        return structure_template

    def aggregate_parameters_via_surrogate(
        self,
        client_weights: List[np.ndarray],
        client_sample_sizes: List[int],
    ) -> np.ndarray:
        """
        Aggregate client parameters using data-weighted convex combination.

        This implements the FedCDH surrogate variable approach:
        Client index acts as implicit latent U, weighted by local data volume.

        Mathematical Form:
            θ_global = Σ_c (n_c / N_total) × θ_c

        Args:
            client_weights: List of parameter arrays from aligned clients
            client_sample_sizes: Local dataset sizes [n_1, n_2, ..., n_C]

        Returns:
            global_weights: Aggregated parameters (same shape as client_weights[0])
        """
        total_samples = sum(client_sample_sizes)

        if total_samples == 0:
            logging.warning("[StructureSync] Zero total samples, using uniform weights")
            normalized_weights = [1.0 / len(client_sample_sizes)] * len(
                client_sample_sizes
            )
        else:
            # Normalize by sample count (FedAvg-style)
            normalized_weights = [n / total_samples for n in client_sample_sizes]

        # Convex combination across structurally aligned parameters
        global_weights = np.zeros_like(client_weights[0])

        for w, alpha in zip(client_weights, normalized_weights):
            global_weights += alpha * w

        logging.info(
            f"[StructureSync] Aggregated {len(client_weights)} clients "
            f"with weights: {[f'{a:.3f}' for a in normalized_weights]}"
        )

        return global_weights

    def validate_structural_alignment(
        self, client_models: List["LocalSPNWrapper"]
    ) -> bool:
        """
        Verify all clients have identical SPN structures.

        Checks:
            1. Same number of parameters
            2. Same model architecture (layer shapes)
            3. Same variable ordering

        Args:
            client_models: List of LocalSPNWrapper instances

        Returns:
            is_aligned: True if all clients structurally identical
        """
        if len(client_models) < 2:
            return True

        # Reference model
        ref_model = client_models[0]
        ref_state = ref_model.model.state_dict()
        ref_keys = sorted(ref_state.keys())

        for i, client in enumerate(client_models[1:], start=1):
            client_state = client.model.state_dict()
            client_keys = sorted(client_state.keys())

            # Check 1: Same parameter names
            if client_keys != ref_keys:
                logging.error(
                    f"[StructureSync] Client {i} has different parameter names"
                )
                return False

            # Check 2: Same parameter shapes
            for key in ref_keys:
                ref_shape = ref_state[key].shape
                client_shape = client_state[key].shape

                if ref_shape != client_shape:
                    logging.error(
                        f"[StructureSync] Client {i} param '{key}' shape mismatch: "
                        f"{ref_shape} vs {client_shape}"
                    )
                    return False

            # Check 3: Variable ordering
            if hasattr(ref_model, "variable_order") and hasattr(
                client, "variable_order"
            ):
                if not torch.equal(ref_model.variable_order, client.variable_order):
                    logging.error(
                        f"[StructureSync] Client {i} has different variable ordering"
                    )
                    return False

        logging.info(
            f"[StructureSync] ✓ All {len(client_models)} clients structurally aligned"
        )
        return True


def broadcast_structure_to_clients(
    num_features: int,
    num_clients: int,
    global_seed: int = 42,
    depth: int = 2,
    num_sums: int = 20,
    **kwargs,
) -> Tuple[Dict, List[int]]:
    """
    Convenience function to generate and broadcast template.

    Args:
        num_features: Dataset dimensionality
        num_clients: Number of federated clients
        global_seed: Structural alignment seed
        depth: SPN depth
        num_sums: Sum nodes per layer
        **kwargs: Additional config parameters

    Returns:
        (template, client_seeds): Template dict and per-client training seeds
    """
    manager = FederatedPCStructureManager(num_features, global_seed)

    template = manager.generate_shared_region_graph(
        depth=depth, num_sums=num_sums, **kwargs
    )

    # Generate different training seeds for each client (data heterogeneity)
    # Structure uses global_seed, training uses client-specific seeds
    np.random.seed(global_seed)
    client_seeds = [np.random.randint(0, 10000) for _ in range(num_clients)]

    return template, client_seeds
