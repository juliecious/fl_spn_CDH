"""
Ensemble SPN implementation for improved stability and performance.

This module provides an ensemble of LocalSPNWrapper models for reduced variance.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from .local import LocalSPNWrapper


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
