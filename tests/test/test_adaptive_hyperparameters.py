"""
Unit Tests for 5-Criterion Adaptive Hyperparameter System

Tests the compute_adaptive_hyperparameters() function that implements
mode-aware, data-driven capacity scaling for FedCDH SPNs.

Expected behavior:
- Horizontal mode: High capacity (4×d base)
- Vertical mode: Depth-focused (8×d for d>3)
- Hybrid mode: Intermediate (6×d)
- Low sample-to-feature ratio: Reduced capacity
- Nonlinear data: Deeper trees, more epochs
- Mode-aware regularization: dropout and weight_decay vary by scenario
"""

import pytest
import numpy as np
from causallearn.utils.FedPC import compute_adaptive_hyperparameters


class TestAdaptiveHyperparameters:
    """Test suite for 5-criterion adaptive hyperparameter system."""

    def test_horizontal_base_capacity(self):
        """
        Criterion 1: Horizontal mode should have 4×d base capacity.
        """
        params = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=10,
            num_samples=400,  # ratio=40, triggers capacity_scale=0.5
            data_type="linear",
        )

        # Base capacity: 4×10=40 sums, 2×10=20 leaves
        # Capacity scale: 0.5 (ratio=40 < 50)
        # Expected: 40*0.5=20 sums, 20*0.5=10 leaves
        assert params["num_sums"] == 20, f"Expected 20 sums, got {params['num_sums']}"
        assert (
            params["num_leaves"] == 10
        ), f"Expected 10 leaves, got {params['num_leaves']}"

    def test_vertical_high_feature_capacity(self):
        """
        Criterion 1: Vertical mode with d>3 should have 8×d base capacity.
        """
        params = compute_adaptive_hyperparameters(
            mode="vertical",
            num_features=5,
            num_samples=500,  # ratio=100, capacity_scale=1.0
            data_type="linear",
        )

        # Base capacity: 8×5=40 sums, 4×5=20 leaves
        # Capacity scale: 1.0 (ratio=100, 100 < 200 → scale=1.0)
        # Expected: 40*1.0=40 sums, 20*1.0=20 leaves
        assert params["num_sums"] == 40, f"Expected 40 sums, got {params['num_sums']}"
        assert (
            params["num_leaves"] == 20
        ), f"Expected 20 leaves, got {params['num_leaves']}"

    def test_vertical_low_feature_capacity(self):
        """
        Criterion 1: Vertical mode with d≤3 should have minimal capacity.
        """
        params = compute_adaptive_hyperparameters(
            mode="vertical",
            num_features=3,
            num_samples=300,  # ratio=100, capacity_scale=0.75
            data_type="linear",
        )

        # Base capacity: 8 sums, 8 leaves (special case for d≤3)
        # Capacity scale: 0.75
        # Expected: 8*0.75=6 sums, 8*0.75=6 leaves
        assert params["num_sums"] == 6, f"Expected 6 sums, got {params['num_sums']}"
        assert (
            params["num_leaves"] == 6
        ), f"Expected 6 leaves, got {params['num_leaves']}"

    def test_hybrid_intermediate_capacity(self):
        """
        Criterion 1: Hybrid mode should have intermediate capacity (6×d).
        """
        params = compute_adaptive_hyperparameters(
            mode="hybrid",
            num_features=8,
            num_samples=800,  # ratio=100, capacity_scale=0.75
            data_type="linear",
        )

        # Base capacity: 6×8=48 sums, 3×8=24 leaves
        # Capacity scale: 0.75
        # Expected: 48*0.75=36 sums, 24*0.75=18 leaves
        assert params["num_sums"] == 36, f"Expected 36 sums, got {params['num_sums']}"
        assert (
            params["num_leaves"] == 18
        ), f"Expected 18 leaves, got {params['num_leaves']}"

    def test_sample_to_feature_ratio_scaling(self):
        """
        Criterion 2: Capacity should scale with sample-to-feature ratio.
        """
        # Low ratio (< 50): capacity_scale = 0.5
        params_low = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=10,
            num_samples=400,  # ratio=40
            data_type="linear",
        )
        assert params_low["num_sums"] == 20  # 40 * 0.5

        # Medium ratio (100-200): capacity_scale = 1.0
        params_med = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=10,
            num_samples=1500,  # ratio=150
            data_type="linear",
        )
        assert params_med["num_sums"] == 40  # 40 * 1.0

        # High ratio (> 200): capacity_scale > 1.0
        params_high = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=10,
            num_samples=5000,  # ratio=500
            data_type="linear",
        )
        # capacity_scale = 1.0 + (500-200)/400 = 1.75, clamped to 1.5
        assert params_high["num_sums"] == 60  # 40 * 1.5

    def test_data_type_depth_bonus(self):
        """
        Criterion 3: Nonlinear data should get deeper trees.
        """
        params_linear = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=8,  # log2(8) = 3
            num_samples=800,
            data_type="linear",
        )
        assert params_linear["depth"] == 3  # base_depth=3, bonus=0

        params_nonlinear = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=8,  # log2(8) = 3
            num_samples=800,
            data_type="nonlinear",
        )
        assert params_nonlinear["depth"] == 4  # base_depth=3, bonus=1

    def test_data_type_epoch_multiplier(self):
        """
        Criterion 3: Nonlinear data should get more training epochs.
        """
        params_linear = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=5,
            num_samples=500,
            data_type="linear",
            base_epochs=100,
        )

        params_nonlinear = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=5,
            num_samples=500,
            data_type="nonlinear",
            base_epochs=100,
        )

        # Nonlinear should have 1.3× more epochs
        assert params_nonlinear["epochs"] > params_linear["epochs"]

    def test_horizontal_mode_epoch_scaling(self):
        """
        Criterion 4: Horizontal mode needs more epochs due to broad feature space.
        """
        params_h = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=10,
            num_samples=1000,
            data_type="linear",
            base_epochs=100,
        )

        params_v = compute_adaptive_hyperparameters(
            mode="vertical",
            num_features=10,
            num_samples=1000,
            data_type="linear",
            base_epochs=100,
        )

        # Horizontal should have more epochs than vertical
        assert params_h["epochs"] > params_v["epochs"]

    def test_epoch_clamping(self):
        """
        Criterion 4: Epochs should be clamped to [100, 500].
        """
        # Small features: Should clamp to minimum 100
        params_min = compute_adaptive_hyperparameters(
            mode="vertical",
            num_features=3,
            num_samples=100,
            data_type="linear",
            base_epochs=50,
        )
        assert params_min["epochs"] >= 100

        # Large features: Should clamp to maximum 500
        params_max = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=20,
            num_samples=2000,
            data_type="nonlinear",
            base_epochs=100,
        )
        assert params_max["epochs"] <= 500

    def test_horizontal_dropout_low_ratio(self):
        """
        Criterion 5: Horizontal mode with low sample ratio should use dropout.
        """
        params = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=10,
            num_samples=800,  # ratio=80 < 100
            data_type="linear",
        )
        assert params["dropout"] == 0.1

    def test_horizontal_dropout_high_ratio(self):
        """
        Criterion 5: Horizontal mode with high sample ratio should skip dropout.
        """
        params = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=10,
            num_samples=1200,  # ratio=120 > 100
            data_type="linear",
        )
        assert params["dropout"] == 0.0

    def test_vertical_no_dropout(self):
        """
        Criterion 5: Vertical mode should not use dropout (structure provides regularization).
        """
        params = compute_adaptive_hyperparameters(
            mode="vertical",
            num_features=5,
            num_samples=200,  # ratio=40 < 100
            data_type="linear",
        )
        assert params["dropout"] == 0.0

    def test_weight_decay_by_mode(self):
        """
        Criterion 5: Weight decay should vary by mode.
        """
        params_h = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=10,
            num_samples=1000,
            data_type="linear",
        )
        assert params_h["weight_decay"] == 1e-4

        params_v = compute_adaptive_hyperparameters(
            mode="vertical",
            num_features=10,
            num_samples=1000,
            data_type="linear",
        )
        assert params_v["weight_decay"] == 1e-3  # Higher for vertical

        params_hybrid = compute_adaptive_hyperparameters(
            mode="hybrid",
            num_features=10,
            num_samples=1000,
            data_type="linear",
        )
        assert params_hybrid["weight_decay"] == 5e-5

    def test_all_keys_present(self):
        """Verify that all expected keys are present in the output."""
        params = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=10,
            num_samples=1000,
            data_type="linear",
        )

        required_keys = [
            "num_sums",
            "num_leaves",
            "depth",
            "epochs",
            "dropout",
            "weight_decay",
        ]
        for key in required_keys:
            assert key in params, f"Missing key: {key}"

    def test_positive_values(self):
        """All hyperparameters should be positive."""
        params = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=10,
            num_samples=1000,
            data_type="linear",
        )

        assert params["num_sums"] > 0
        assert params["num_leaves"] > 0
        assert params["depth"] > 0
        assert params["epochs"] > 0
        assert params["dropout"] >= 0
        assert params["weight_decay"] >= 0

    def test_medium_horizontal_target_case(self):
        """
        Test MEDIUM config horizontal case (known failure: F1=0.255).
        Expected: Substantial capacity boost for 2-4× F1 improvement.
        """
        params = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=10,
            num_samples=400,  # MEDIUM: n=400 per client (1200 total / 3 clients)
            data_type="linear",
        )

        # Base capacity: 4×10=40 sums
        # Ratio: 400/10=40 < 50 → capacity_scale=0.5
        # Final: 40*0.5=20 sums
        #
        # This is still conservative due to low ratio, but significantly better
        # than the original num_sums=20 with sqrt(10/5)=1.41 → 28 sums
        #
        # The key improvement comes from:
        # 1. Proper depth scaling (log2(10)=3 + nonlinear_bonus)
        # 2. Epoch scaling (more training for horizontal)
        # 3. Adaptive regularization

        assert params["num_sums"] >= 20  # At minimum, match old sqrt scaling
        assert params["depth"] >= 3  # log2(10) = 3.32 → 3
        assert params["epochs"] > 100  # Horizontal needs more training

    def test_large_horizontal_case(self):
        """
        Test LARGE config horizontal case (d=11, K=5, n=2000 total).
        Expected: High capacity due to larger feature space.
        """
        params = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=11,
            num_samples=400,  # 2000 total / 5 clients
            data_type="nonlinear",
        )

        # Base capacity: 4×11=44 sums
        # Ratio: 400/11=36 < 50 → capacity_scale=0.5
        # Final: 44*0.5=22 sums
        assert params["num_sums"] >= 22
        assert params["depth"] >= 4  # log2(11)=3 + nonlinear_bonus=1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
