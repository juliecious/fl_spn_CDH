import unittest
import numpy as np
import sys
import os

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH
from causallearn.utils.data_utils import (
    set_random_seed,
    simulate_dag,
    my_simulate_linear_gaussian,
)


class TestFedCDHPipeline(unittest.TestCase):
    def test_fedcdh_smoke_horizontal(self):
        """
        Runs a tiny FedCDH instance to ensure no crashes.
        """

        class Args:
            def __init__(self):
                self.n = 50
                self.d = 5
                self.K = 2
                self.scenario = "horizontal"
                self.ci_method = "spn"
                self.model_type = "linear"  # synthetic random
                self.ablation_orientation = "mi_hybrid"

        args = Args()
        set_random_seed(0)

        # Generate Data
        true_dag = simulate_dag(args.d, args.d, "ER")
        X_global, c_indx = my_simulate_linear_gaussian(
            true_dag, args.K, args.n * args.K, "gauss"
        )
        c_indx = np.repeat(np.arange(args.K), args.n).reshape(-1, 1)
        X_global = (X_global - X_global.mean(0)) / (X_global.std(0) + 1e-6)
        X_splits = np.array_split(X_global, args.K)

        # Run
        runner = FedCDH(args)
        result = runner.fit(X_splits, c_indx, true_dag)

        # Assertions
        self.assertIsNotNone(result)
        self.assertIn("f1_skeleton", result)
        print(f"Smoke Test Result: {result}")


if __name__ == "__main__":
    unittest.main()
