"""
Bug 1 & 2 Fix: Proper Client Initialization with Structural Sync.

This module provides utilities to initialize federated clients with:
1. Shared structural template (deterministic architecture)
2. Independent parameter initialization (heterogeneous learning)

Reference: Dry Run Analysis - Bugs 1 & 2
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Tuple
from ..core.local import LocalSPNWrapper


def initialize_heterogeneous_clients(
    client_datasets: List[np.ndarray],
    template: Dict,
    client_seeds: List[int],
    device: str = "cpu",
    verbose: bool = False,
) -> List[LocalSPNWrapper]:
    """
    Initialize clients with shared structure but unique parameters.

    Fixes two critical bugs:
    1. Template structure parameter is now properly extracted and passed
    2. Each client gets unique torch seed BEFORE model creation

    Args:
        client_datasets: List of local datasets per client
        template: Structure template from broadcast_structure_to_clients()
        client_seeds: Unique random seeds for parameter initialization
        device: CPU or CUDA
        verbose: Print client info

    Returns:
        clients: List of LocalSPNWrapper instances

    Example:
        >>> template, seeds = broadcast_structure_to_clients(8, 3, 42)
        >>> clients = initialize_heterogeneous_clients(
        ...     [data0, data1, data2], template, seeds
        ... )
        >>> # Verify clients have DIFFERENT parameters but SAME structure
        >>> assert clients[0].model.state_dict().keys() == clients[1].model.state_dict().keys()
        >>> w0 = clients[0].model.state_dict()['layer.0.weight']
        >>> w1 = clients[1].model.state_dict()['layer.0.weight']
        >>> assert not torch.allclose(w0, w1)  # Different parameters!
    """
    # Extract configuration from template
    num_features = template["num_features"]
    structure_type = template["config"].get("structure", "poon-domingos")
    depth = template["config"].get("depth", 3)
    num_sums = template["config"].get("num_sums", 20)
    num_leaves = template["config"].get("num_leaves", 20)
    num_repetitions = template["config"].get("num_repetitions", 10)

    if verbose:
        logging.info(f"[ClientInit] Initializing {len(client_datasets)} clients")
        logging.info(f"  Structure: {structure_type}, Depth: {depth}")
        logging.info(f"  Shared template seed: {template['seed']}")

    clients = []

    for client_id, (client_data, client_seed) in enumerate(
        zip(client_datasets, client_seeds)
    ):
        # BUG FIX: Set unique PyTorch seed BEFORE model instantiation
        # This ensures different initial parameters for each client
        torch.manual_seed(client_seed)
        np.random.seed(client_seed)

        if verbose:
            logging.info(
                f"[Client {client_id}] "
                f"Samples: {len(client_data)}, Seed: {client_seed}"
            )

        # BUG FIX: Pass structure parameter explicitly
        # Previously, LocalSPNWrapper ignored template and used hardcoded "top-down"
        client = LocalSPNWrapper(
            num_features=num_features,
            seed=client_seed,  # CRITICAL: Unique seed per client
            structure=structure_type,  # CRITICAL: Use template structure
            depth=depth,
            num_sums=num_sums,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
            device=device,
        )

        clients.append(client)

        # Verify parameters are unique (debugging)
        if verbose and client_id > 0:
            # Compare with first client
            ref_params = clients[0].model.state_dict()
            client_params = client.model.state_dict()

            # Check first weight matrix
            first_key = list(ref_params.keys())[0]
            ref_weight = ref_params[first_key]
            client_weight = client_params[first_key]

            param_diff = torch.norm(ref_weight - client_weight).item()
            logging.debug(
                f"[Client {client_id}] Param difference from Client 0: {param_diff:.4f}"
            )

            if param_diff < 0.1:
                logging.warning(
                    f"[Client {client_id}] WARNING: Parameters too similar to Client 0! "
                    f"Seed may not be working correctly."
                )

    return clients


def train_clients_locally(
    clients: List[LocalSPNWrapper],
    client_datasets: List[np.ndarray],
    epochs: int = 50,
    lr: float = 0.005,
    verbose: bool = False,
) -> List[float]:
    """
    Train clients on their local datasets.

    Args:
        clients: List of initialized LocalSPNWrapper instances
        client_datasets: List of local datasets
        epochs: Training epochs per client
        lr: Learning rate
        verbose: Print training progress

    Returns:
        final_lls: List of final log-likelihoods per client
    """
    final_lls = []

    for client_id, (client, client_data) in enumerate(zip(clients, client_datasets)):
        if verbose:
            logging.info(
                f"[Training] Client {client_id}: {len(client_data)} samples, "
                f"{epochs} epochs"
            )

        final_ll = client.train_local(client_data, epochs=epochs, lr=lr)
        final_lls.append(final_ll)

        if verbose:
            logging.info(f"[Training] Client {client_id}: Final LL = {final_ll:.4f}")

    return final_lls


def validate_structural_alignment_detailed(
    clients: List[LocalSPNWrapper], verbose: bool = True
) -> bool:
    """
    Detailed validation that clients have identical structures.

    Checks:
    1. Same parameter names
    2. Same parameter shapes
    3. DIFFERENT parameter values (heterogeneous learning)

    Args:
        clients: List of client models
        verbose: Print validation details

    Returns:
        is_valid: True if all checks pass
    """
    if len(clients) < 2:
        return True

    ref_client = clients[0]
    ref_state = ref_client.model.state_dict()
    ref_keys = sorted(ref_state.keys())

    if verbose:
        logging.info(
            f"[Validation] Checking structural alignment for {len(clients)} clients"
        )

    for client_id, client in enumerate(clients[1:], start=1):
        client_state = client.model.state_dict()
        client_keys = sorted(client_state.keys())

        # Check 1: Same parameter names
        if client_keys != ref_keys:
            if verbose:
                logging.error(
                    f"[Validation] Client {client_id} has different parameter names!"
                )
            return False

        # Check 2: Same parameter shapes
        for key in ref_keys:
            ref_shape = ref_state[key].shape
            client_shape = client_state[key].shape

            if ref_shape != client_shape:
                if verbose:
                    logging.error(
                        f"[Validation] Client {client_id} param '{key}' shape mismatch: "
                        f"{ref_shape} vs {client_shape}"
                    )
                return False

        # Check 3: DIFFERENT parameter values (heterogeneity check)
        total_diff = 0.0
        num_params = 0

        for key in ref_keys:
            if "weight" in key or "bias" in key:
                ref_param = ref_state[key].flatten()
                client_param = client_state[key].flatten()

                diff = torch.norm(ref_param - client_param).item()
                total_diff += diff
                num_params += 1

        avg_diff = total_diff / max(num_params, 1)

        if verbose:
            logging.info(
                f"[Validation] Client {client_id}: Avg param difference = {avg_diff:.4f}"
            )

        if avg_diff < 0.1:
            if verbose:
                logging.warning(
                    f"[Validation] WARNING: Client {client_id} parameters too similar "
                    f"to reference (avg_diff={avg_diff:.4f}). "
                    f"Clients may not have heterogeneous initialization!"
                )

    if verbose:
        logging.info("[Validation] ✓ All clients structurally aligned")

    return True
