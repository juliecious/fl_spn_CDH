import unittest
import torch
import numpy as np
import sys
import os

# Ensure we can import modules from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fed_spn.structure import FederatedRatSPN
from data.synthetic import generate_synthetic_data, create_hybrid_splits


class TestFederatedSPN(unittest.TestCase):
    def setUp(self):
        # Parameters matching FedCDH test scales
        self.d = 6
        self.n_samples = 1000  # Increased from 100 to ensure convergence
        self.n_clients = 3
        self.seed = 100
        self.device = "cpu"  # Keep simple for CI/Test

        # 1. Generate Synthetic Data (Linear SEM)
        raw_data, self.ground_truth_dag = generate_synthetic_data(
            n_samples=self.n_samples, d=self.d, seed=self.seed, model_type="linear"
        )

        # 2. Create Hybrid Splits
        # clients_splits is list of (data_tensor, indices_array)
        self.clients_splits = create_hybrid_splits(
            raw_data, self.n_clients, seed=self.seed
        )

        # Convert to Torch
        self.clients_splits = [
            (torch.FloatTensor(d), idx) for d, idx in self.clients_splits
        ]

    def test_structure_synchronization(self):
        """
        Verify that Client A and Client B generate the same SPN structure
        if provided the same seed.
        """
        model_a = FederatedRatSPN(self.d, seed=42)
        model_b = FederatedRatSPN(self.d, seed=42)
        model_c = FederatedRatSPN(self.d, seed=99)  # Different seed

        # Check weights are identical initially
        w_a = model_a.state_dict()
        w_b = model_b.state_dict()

        for k in w_a:
            self.assertTrue(torch.equal(w_a[k], w_b[k]), f"Mismatch in param {k}")

        # Check model_c is different
        # Note: weights might be randomly initialized differently,
        # but crucial is that structure (param shapes) match.
        # Here we check strict equality which implies same init.
        w_c = model_c.state_dict()
        is_diff = any(not torch.equal(w_a[k], w_c[k]) for k in w_a)
        self.assertTrue(is_diff, "Different seeds should produce different inits")

    def test_hybrid_training_step(self):
        """
        Test that training doesn't crash on masked data and
        that gradients are zero for missing features.
        """
        model = FederatedRatSPN(self.d, seed=self.seed)

        # Take Client 0
        data, indices = self.clients_splits[0]
        missing_indices = list(set(range(self.d)) - set(indices))

        # Run one step
        initial_loss = model.train_local_step(data, indices)
        self.assertFalse(np.isnan(initial_loss), "Loss became NaN")

        # Check Gradient Masking
        # Retrieve a known leaf parameter
        for name, param in model.model.named_parameters():
            if "leaf" in name and param.grad is not None:
                if param.shape[0] == self.d:
                    # Check if gradients for missing indices are 0
                    grad_slice = param.grad[missing_indices, ...]
                    self.assertTrue(
                        torch.all(grad_slice == 0),
                        f"Gradient leak detected in {name} for missing vars!",
                    )

    def test_federated_convergence(self):
        """
        Simulate a mini Federated Learning loop and check if loss decreases.
        """
        global_model = FederatedRatSPN(self.d, seed=self.seed)

        initial_loss = 0
        final_loss = 0

        # Helper to average weights
        def fed_avg(weights_list):
            avg_weights = {}
            for k in weights_list[0].keys():
                avg_weights[k] = torch.stack([w[k] for w in weights_list]).mean(dim=0)
            return avg_weights

        print("\n--- Starting Federated Training Test ---")

        # 5 Rounds
        for round_idx in range(5):
            local_weights = []
            round_losses = []

            # Broadcast
            global_w = global_model.get_weights()

            for client_idx in range(self.n_clients):
                data, indices = self.clients_splits[client_idx]

                # Client Local Update
                client_model = FederatedRatSPN(self.d, seed=self.seed)
                client_model.set_weights(global_w)

                # Train 5 epochs locally
                c_loss = 0
                for _ in range(5):
                    c_loss = client_model.train_local_step(data, indices)

                local_weights.append(client_model.get_weights())
                round_losses.append(c_loss)

            # Aggregate
            new_global_w = fed_avg(local_weights)
            global_model.set_weights(new_global_w)

            avg_round_loss = np.mean(round_losses)
            print(f"Round {round_idx} Avg Loss: {avg_round_loss:.4f}")

            if round_idx == 0:
                initial_loss = avg_round_loss
            final_loss = avg_round_loss

        self.assertLess(final_loss, initial_loss, "Model failed to converge/learn")


if __name__ == "__main__":
    unittest.main()
