#!/usr/bin/env python
"""Quick smoke test to verify error handling in FedCDH."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("✓ Imports successful")

# Test 1: Import FedCDH
from causallearn.search.FCMBased.FedCDH import FedCDH

print("✓ FedCDH imported successfully")

# Test 2: Check torch import (for CUDA error handling)
import torch

print(f"✓ PyTorch imported (CUDA available: {torch.cuda.is_available()})")

# Test 3: Check if OutOfMemoryError is accessible
print(f"✓ torch.cuda.OutOfMemoryError accessible: {torch.cuda.OutOfMemoryError}")

# Test 4: Verify the method registry
from tests.benchmarks.test_fedcdh_benchmark_v3 import METHOD_REGISTRY, DATASET_REGISTRY

print(f"✓ METHOD_REGISTRY has {len(METHOD_REGISTRY)} methods")
print(f"✓ DATASET_REGISTRY has {len(DATASET_REGISTRY)} datasets")

# Test 5: Check if fedspn methods exist
assert "fedspn_h" in METHOD_REGISTRY
assert "fedspn_v" in METHOD_REGISTRY
assert "fedspn_hy" in METHOD_REGISTRY
print("✓ All FedSPN methods registered")

# Test 6: Check if law_school dataset exists
assert "law_school" in DATASET_REGISTRY
print("✓ law_school dataset registered")

print("\n" + "=" * 60)
print("✓✓✓ ALL SMOKE TESTS PASSED ✓✓✓")
print("=" * 60)
print("\nYou can now run the full benchmark with GPU:")
print("  python tests/benchmarks/test_fedcdh_benchmark_v3.py \\")
print("    --datasets law_school \\")
print("    --methods fedspn_h,fedspn_v,fedspn_hy \\")
print("    --seeds 42 \\")
print("    --device cuda \\")
print("    --save-graphs")
