#!/usr/bin/env python
"""
Test script for all benchmark datasets: DREAM4, Asia, Alarm.

This script verifies that all dataset loaders work correctly and prints
comprehensive statistics about each benchmark.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tests.utils.dream_loader import (
    load_dream4_network,
    generate_dream4_data,
    get_dream4_statistics,
)
from tests.utils.bayesian_network_loaders import (
    generate_asia_data,
    generate_alarm_data,
    get_network_info,
)


def test_dream4_networks():
    """Test all DREAM4 networks."""
    print("\n" + "=" * 80)
    print("TESTING DREAM4 NETWORKS")
    print("=" * 80)

    get_dream4_statistics()

    # Test data generation for each network
    print("\n" + "-" * 80)
    print("Testing data generation...")
    print("-" * 80)

    for net_id in range(1, 6):
        for mode in ["linear", "nonlinear"]:
            X, B, names = generate_dream4_data(
                network_id=net_id, n_samples=500, mode=mode, seed=42
            )

            # Verify data properties
            assert X.shape == (500, 10), f"Wrong data shape for network {net_id}"
            assert B.shape == (10, 10), f"Wrong adjacency shape for network {net_id}"
            assert len(names) == 10, f"Wrong number of names for network {net_id}"

            # Check standardization
            assert abs(X.mean()) < 0.1, f"Data not centered for network {net_id}"
            assert (
                abs(X.std() - 1.0) < 0.2
            ), f"Data not standardized for network {net_id}"

            # Check DAG property (no cycles)
            assert np.allclose(B, B.astype(int)), "Adjacency should be binary"
            assert np.all(B >= 0), "Adjacency should be non-negative"

            print(f"  ✓ Network {net_id} ({mode}): {X.shape}, {int(B.sum())} edges")

    print("\n✅ All DREAM4 networks passed tests!")


def test_asia_network():
    """Test Asia network."""
    print("\n" + "=" * 80)
    print("TESTING ASIA NETWORK")
    print("=" * 80)

    get_network_info("asia")

    print("\n" + "-" * 80)
    print("Testing data generation...")
    print("-" * 80)

    for mode in ["linear", "nonlinear"]:
        X, B, names = generate_asia_data(n_samples=1000, mode=mode, seed=42)

        # Verify
        assert X.shape == (1000, 8), "Wrong data shape"
        assert B.shape == (8, 8), "Wrong adjacency shape"
        assert len(names) == 8, "Wrong number of names"
        assert (
            int(B.sum()) == 8
        ), f"Wrong number of edges: expected 8, got {int(B.sum())}"

        # Check standardization
        assert abs(X.mean()) < 0.1, "Data not centered"
        assert abs(X.std() - 1.0) < 0.2, "Data not standardized"

        print(f"  ✓ Asia ({mode}): {X.shape}, {int(B.sum())} edges")
        print(f"    Variables: {', '.join(names)}")

    print("\n✅ Asia network passed tests!")


def test_alarm_network():
    """Test Alarm network."""
    print("\n" + "=" * 80)
    print("TESTING ALARM NETWORK")
    print("=" * 80)

    get_network_info("alarm")

    print("\n" + "-" * 80)
    print("Testing data generation...")
    print("-" * 80)

    for mode in ["linear", "nonlinear"]:
        X, B, names = generate_alarm_data(n_samples=1000, mode=mode, seed=42)

        # Verify
        assert X.shape == (1000, 37), "Wrong data shape"
        assert B.shape == (37, 37), "Wrong adjacency shape"
        assert len(names) == 37, "Wrong number of names"

        # Check standardization
        assert abs(X.mean()) < 0.1, "Data not centered"
        assert abs(X.std() - 1.0) < 0.2, "Data not standardized"

        print(f"  ✓ Alarm ({mode}): {X.shape}, {int(B.sum())} edges")

    print("\n✅ Alarm network passed tests!")


def compare_all_benchmarks():
    """Compare all benchmarks side-by-side."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON SUMMARY")
    print("=" * 80)

    benchmarks = []

    # DREAM4 networks
    for net_id in range(1, 6):
        B, names = load_dream4_network(net_id)
        benchmarks.append(
            {
                "name": f"DREAM4 Net{net_id}",
                "nodes": len(names),
                "edges": int(B.sum()),
                "density": B.sum() / (len(names) * (len(names) - 1)),
            }
        )

    # Asia
    X, B, names = generate_asia_data(100)
    benchmarks.append(
        {
            "name": "Asia",
            "nodes": len(names),
            "edges": int(B.sum()),
            "density": B.sum() / (len(names) * (len(names) - 1)),
        }
    )

    # Alarm
    X, B, names = generate_alarm_data(100)
    benchmarks.append(
        {
            "name": "Alarm",
            "nodes": len(names),
            "edges": int(B.sum()),
            "density": B.sum() / (len(names) * (len(names) - 1)),
        }
    )

    # Print table
    print(
        f"\n{'Dataset':<20} {'Nodes':>6} {'Edges':>6} {'Density':>8} {'Difficulty':<15}"
    )
    print("-" * 80)

    for b in benchmarks:
        # Assign difficulty based on size and density
        if b["nodes"] <= 10:
            difficulty = "Easy-Medium"
        elif b["nodes"] <= 20:
            difficulty = "Medium"
        else:
            difficulty = "Medium-Hard"

        print(
            f"{b['name']:<20} {b['nodes']:>6} {b['edges']:>6} {b['density']:>8.4f} {difficulty:<15}"
        )

    print("\n" + "=" * 80)


def generate_example_datasets():
    """Generate example datasets for immediate use."""
    print("\n" + "=" * 80)
    print("GENERATING EXAMPLE DATASETS")
    print("=" * 80)

    output_dir = Path(__file__).parent.parent / "data" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    datasets_generated = []

    # Generate DREAM4 Network 1 (representative)
    X, B, names = generate_dream4_data(
        network_id=1, n_samples=1000, mode="linear", seed=42
    )
    df = pd.DataFrame(X, columns=names)
    data_file = output_dir / "dream4_net1_linear_n1000.csv"
    df.to_csv(data_file, index=False)
    datasets_generated.append(str(data_file))

    # Generate Asia
    X, B, names = generate_asia_data(n_samples=1000, mode="linear", seed=42)
    df = pd.DataFrame(X, columns=names)
    data_file = output_dir / "asia_linear_n1000.csv"
    df.to_csv(data_file, index=False)
    datasets_generated.append(str(data_file))

    # Generate Alarm
    X, B, names = generate_alarm_data(n_samples=1000, mode="linear", seed=42)
    df = pd.DataFrame(X, columns=names)
    data_file = output_dir / "alarm_linear_n1000.csv"
    df.to_csv(data_file, index=False)
    datasets_generated.append(str(data_file))

    print("\n✅ Generated example datasets:")
    for f in datasets_generated:
        print(f"  - {f}")

    print(f"\n📁 All files saved to: {output_dir}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("BENCHMARK DATASET TEST SUITE")
    print("=" * 80)
    print("\nThis script tests DREAM4, Asia, and Alarm benchmark loaders.")

    try:
        # Run tests
        test_dream4_networks()
        test_asia_network()
        test_alarm_network()

        # Comparison
        compare_all_benchmarks()

        # Generate example datasets
        generate_example_datasets()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nYou now have access to the following benchmarks:")
        print("  • DREAM4: 5 networks × 10 nodes each (gene regulatory networks)")
        print("  • Asia: 8 nodes (medical diagnosis)")
        print("  • Alarm: 37 nodes (medical monitoring)")
        print("\nUsage examples:")
        print("  from tests.utils.dream_loader import generate_dream4_data")
        print("  X, B, names = generate_dream4_data(network_id=1, n_samples=1000)")
        print("\n  from tests.utils.bayesian_network_loaders import generate_asia_data")
        print("  X, B, names = generate_asia_data(n_samples=1000)")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
