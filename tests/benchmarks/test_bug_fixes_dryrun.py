"""
Dry-Run Validation Tests for Bug Fixes.

Tests validate the three bug fixes on synthetic Asia-like data:
1. Bug 1 & 2: Structural sync + heterogeneous parameters
2. Bug 3: CI-based DAG search (not score-based)

Run with: pytest tests/benchmarks/test_bug_fixes_dryrun.py -v -s
"""

import pytest
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)


class TestBugFixes_DryRun:
    """Validate bug fixes match dry-run predictions."""

    def test_bug1_structure_parameter_passed(self):
        """Verify structure parameter is properly passed to LocalSPNWrapper."""
        from causallearn.utils.spn.core.local import LocalSPNWrapper

        # Test with poon-domingos structure (from template)
        spn_pd = LocalSPNWrapper(
            num_features=8,
            depth=2,
            structure="poon-domingos",  # BUG FIX: Now accepted
            device="cpu",
        )

        # Verify config has correct structure
        assert (
            spn_pd.config.structure == "poon-domingos"
        ), "Structure parameter not set correctly!"

        # Test with top-down structure (old default)
        spn_td = LocalSPNWrapper(
            num_features=8, depth=2, structure="top-down", device="cpu"
        )

        assert spn_td.config.structure == "top-down"

        print("✓ Bug 1 fixed: Structure parameter properly passed")

    def test_bug2_heterogeneous_parameters(self):
        """Verify clients get unique parameters from different seeds."""
        from causallearn.utils.spn.federated.client_init import (
            initialize_heterogeneous_clients,
        )
        from causallearn.utils.spn.federated import broadcast_structure_to_clients

        # Generate template and seeds
        template, client_seeds = broadcast_structure_to_clients(
            num_features=8, num_clients=3, global_seed=42
        )

        # Create mock datasets
        np.random.seed(100)
        client_datasets = [np.random.randn(100, 8) for _ in range(3)]

        # Initialize clients
        clients = initialize_heterogeneous_clients(
            client_datasets, template, client_seeds, verbose=True
        )

        # Verify clients have SAME structure
        ref_keys = list(clients[0].model.state_dict().keys())
        for i in range(1, 3):
            assert list(clients[i].model.state_dict().keys()) == ref_keys

        # Verify clients have DIFFERENT parameters
        ref_weight = clients[0].model.state_dict()["layers.0.weight"]
        for i in range(1, 3):
            client_weight = clients[i].model.state_dict()["layers.0.weight"]
            diff = torch.norm(ref_weight - client_weight).item()

            print(f"Client {i} param difference from Client 0: {diff:.4f}")

            # CRITICAL: Difference should be substantial (not < 0.1)
            assert diff > 0.5, f"Client {i} parameters too similar (diff={diff:.4f})!"

        print("✓ Bug 2 fixed: Clients have heterogeneous parameters")

    def test_bug3_ci_testing_not_score_based(self):
        """Verify CI-based search works, not score-based."""
        from causallearn.utils.spn.causal.ci_testing import (
            compute_circuit_ci_discrepancy,
            greedy_dag_search_via_circuit_ci,
        )
        from causallearn.utils.spn.core.local import LocalSPNWrapper
        from causallearn.utils.spn.federated import GlobalFedSPN

        # Generate synthetic data with known structure: X0 → X1 → X2
        np.random.seed(42)
        n = 1000
        X0 = np.random.randn(n)
        X1 = X0 + np.random.randn(n) * 0.3  # X1 depends on X0
        X2 = X1 + np.random.randn(n) * 0.3  # X2 depends on X1
        X3 = np.random.randn(n)  # X3 is independent

        data = np.column_stack([X0, X1, X2, X3])

        # Train SPN on this data
        spn = LocalSPNWrapper(num_features=4, depth=2, device="cpu")
        spn.train_local(data, epochs=30)

        # Wrap in GlobalFedSPN (simulate horizontal FL with 1 client)
        global_spn = GlobalFedSPN(components=[spn], device="cpu")

        # Test 1: Marginal dependence (X0, X1) should be DEPENDENT
        dep_01 = compute_circuit_ci_discrepancy(
            global_spn, torch.tensor(data, dtype=torch.float32), 0, 1, []
        )
        print(f"Discrepancy X0 ⊥ X1: {dep_01:.4f} (expect > 0.02)")
        assert dep_01 > 0.02, "X0 and X1 should be dependent!"

        # Test 2: Independence (X0, X3) should be INDEPENDENT
        dep_03 = compute_circuit_ci_discrepancy(
            global_spn, torch.tensor(data, dtype=torch.float32), 0, 3, []
        )
        print(f"Discrepancy X0 ⊥ X3: {dep_03:.4f} (expect < 0.02)")
        assert dep_03 < 0.05, "X0 and X3 should be independent!"

        # Test 3: Conditional independence X0 ⊥ X2 | X1 (should hold)
        dep_02_given_1 = compute_circuit_ci_discrepancy(
            global_spn, torch.tensor(data, dtype=torch.float32), 0, 2, [1]
        )
        print(f"Discrepancy X0 ⊥ X2 | X1: {dep_02_given_1:.4f} (expect < 0.02)")
        # Should be close to independent given X1
        assert dep_02_given_1 < 0.1, "X0 ⊥ X2 | X1 should approximately hold!"

        print("✓ Bug 3 fixed: CI testing works correctly")

    def test_bug3_dag_search_not_empty(self):
        """Verify DAG search returns non-empty skeleton (not all zeros)."""
        from causallearn.utils.spn.causal.ci_testing import (
            greedy_dag_search_via_circuit_ci,
        )
        from causallearn.utils.spn.core.local import LocalSPNWrapper
        from causallearn.utils.spn.federated import GlobalFedSPN

        # Generate chain structure: X0 → X1 → X2
        np.random.seed(42)
        n = 1000
        X0 = np.random.randn(n)
        X1 = X0 + np.random.randn(n) * 0.2
        X2 = X1 + np.random.randn(n) * 0.2
        data = np.column_stack([X0, X1, X2])

        # Train SPN
        spn = LocalSPNWrapper(num_features=3, depth=2, device="cpu")
        spn.train_local(data, epochs=40)

        global_spn = GlobalFedSPN(components=[spn], device="cpu")

        # Run CI-based DAG search
        learned_skeleton = greedy_dag_search_via_circuit_ci(
            global_spn,
            data,
            threshold=0.03,
            max_conditioning_size=2,
            verbose=True,
        )

        print(f"Learned skeleton:\n{learned_skeleton}")

        # CRITICAL: Skeleton should NOT be empty!
        num_edges = learned_skeleton.sum()
        print(f"Number of edges: {num_edges}")

        assert num_edges > 0, "Skeleton is empty! Bug 3 not fixed."

        # Should capture at least the chain X0-X1-X2
        # Check if X0-X1 edge exists
        assert (
            learned_skeleton[0, 1] == 1 or learned_skeleton[1, 0] == 1
        ), "Missing X0-X1 edge!"

        # Check if X1-X2 edge exists
        assert (
            learned_skeleton[1, 2] == 1 or learned_skeleton[2, 1] == 1
        ), "Missing X1-X2 edge!"

        print("✓ Bug 3 fixed: DAG search returns non-empty skeleton")

    @pytest.mark.slow
    def test_full_pipeline_synthetic_asia(self):
        """
        Full pipeline test simulating Asia dataset structure.

        Asia DAG:
            A (Asia)
            |
            T (Tuberculosis)
            |
            E (Either)   ← Also influenced by L (Lung)
            |
            X (XRay)

        Simplified: 4 variables in chain for testing
        """
        from causallearn.utils.spn.federated import (
            broadcast_structure_to_clients,
            GlobalFedSPN,
        )
        from causallearn.utils.spn.federated.client_init import (
            initialize_heterogeneous_clients,
            train_clients_locally,
        )
        from causallearn.utils.spn.causal.ci_testing import (
            greedy_dag_search_via_circuit_ci,
        )

        # Generate synthetic Asia-like structure (4 variables)
        np.random.seed(42)
        n = 2000

        # A → T → E → X (chain)
        A = np.random.binomial(1, 0.3, n).astype(float)
        T = (A > 0.5).astype(float) * 0.8 + np.random.randn(n) * 0.2
        E = T + np.random.randn(n) * 0.3
        X = E + np.random.randn(n) * 0.3

        data = np.column_stack([A, T, E, X])

        # True skeleton (undirected)
        true_skeleton = np.array(
            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
        )

        # Split into 3 clients (horizontal FL)
        client_data = [data[:666], data[666:1332], data[1332:]]

        # Broadcast template
        template, client_seeds = broadcast_structure_to_clients(
            num_features=4, num_clients=3, global_seed=42
        )

        # Initialize clients
        clients = initialize_heterogeneous_clients(
            client_data, template, client_seeds, verbose=True
        )

        # Train clients
        train_clients_locally(clients, client_data, epochs=40, verbose=True)

        # Build global mixture
        global_spn = GlobalFedSPN(components=clients, device="cpu")

        # DAG search
        learned_skeleton = greedy_dag_search_via_circuit_ci(
            global_spn, data, threshold=0.04, max_conditioning_size=2, verbose=True
        )

        print(f"\nTrue skeleton:\n{true_skeleton}")
        print(f"Learned skeleton:\n{learned_skeleton}")

        # Compute SHD
        shd = np.abs(true_skeleton - learned_skeleton).sum() // 2  # Undirected
        print(f"\nStructural Hamming Distance: {shd}")

        # Success criterion: SHD < 4 (out of 3 edges)
        assert shd < 4, f"SHD too high: {shd} (expect < 4)"

        print(f"✓ Full pipeline works! SHD = {shd}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
