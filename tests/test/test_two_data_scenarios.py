"""
Test two data generation scenarios:
1. simulate_dag=ER + my_simulate_linear_gaussian + sem_type='gauss'
2. simulate_dag=ER + my_simulate_general_hetero + sem_type='gauss'
"""

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
from test_spn_methods_smoke import run_smoke_test

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def main():
    """Run smoke tests for two data generation scenarios across all three SPN scenarios."""
    logging.info("\n" + "=" * 80)
    logging.info("TWO DATA GENERATION SCENARIOS TEST")
    logging.info(
        "Scenario 1: Linear Gaussian (my_simulate_linear_gaussian + sem_type='gauss')"
    )
    logging.info(
        "Scenario 2: Nonlinear Gaussian (my_simulate_general_hetero + sem_type='gauss')"
    )
    logging.info("Both using: simulate_dag with graph_type='ER'")
    logging.info("=" * 80 + "\n")

    # Test parameters (small for fast smoke test)
    test_params = {
        "d": 5,  # Small number of variables
        "K": 2,  # Minimal number of clients
        "n": 200,  # Small sample size
        "epochs": 20,  # Reduced epochs for speed
        "seed": 42,
    }

    scenarios = ["horizontal", "vertical", "hybrid"]
    data_configs = [
        {"data_type": "linear", "sem_type": "gauss", "name": "Linear Gaussian"},
        {"data_type": "nonlinear", "sem_type": "gauss", "name": "Nonlinear Gaussian"},
    ]

    all_results = []

    # Test each combination
    for data_config in data_configs:
        logging.info("\n" + "=" * 80)
        logging.info(f"DATA GENERATION: {data_config['name']}")
        logging.info(
            f"  - Function: {'my_simulate_linear_gaussian' if data_config['data_type'] == 'linear' else 'my_simulate_general_hetero'}"
        )
        logging.info(f"  - SEM type: {data_config['sem_type']}")
        logging.info(f"  - Graph type: ER (Erdős-Rényi)")
        logging.info("=" * 80 + "\n")

        for scenario in scenarios:
            result = run_smoke_test(
                scenario=scenario,
                data_type=data_config["data_type"],
                sem_type=data_config["sem_type"],
                **test_params,
            )
            result["data_config"] = data_config["name"]
            all_results.append(result)

    # Print comprehensive summary
    logging.info("\n" + "=" * 80)
    logging.info("COMPREHENSIVE SUMMARY")
    logging.info("=" * 80)
    logging.info("")

    for data_config in data_configs:
        logging.info(f"{data_config['name']}:")
        config_results = [
            r for r in all_results if r["data_config"] == data_config["name"]
        ]

        for result in config_results:
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            logging.info(
                f"  {result['scenario'].upper():12s} {status:8s}  "
                f"Time: {result['total_time']:5.1f}s  "
                f"Skeleton F1: {result['skeleton_f1']:.3f}  "
                f"DAG F1: {result['dag_f1']:.3f}"
                if result["dag_f1"] is not None
                else f"DAG F1: None"
            )

            if not result["passed"]:
                for reason in result["reasons"]:
                    logging.info(f"    └─ {reason}")
        logging.info("")

    logging.info("=" * 80)

    # Check if all passed
    all_passed = all(r["passed"] for r in all_results)
    if all_passed:
        logging.info("\n✓ All tests PASSED!")
        return 0
    else:
        logging.info("\n✗ Some tests FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
