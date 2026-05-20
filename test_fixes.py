#!/usr/bin/env python
"""Quick test to verify the two critical fixes."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("Testing Fix 1: CPU Fallback Device Mismatch")
print("Testing Fix 2: Hybrid Memory Usage Reduction")
print()

# Test that imports work
from causallearn.search.FCMBased.FedCDH import FedCDH

print("✓ FedCDH imported successfully")

# Test that the code changes are present
import inspect

source = inspect.getsource(FedCDH.fit)

# Check for device move after CPU fallback
if "moved back to" in source:
    print("✓ Fix 1: CPU fallback device move code present")
else:
    print("✗ Fix 1: CPU fallback device move code MISSING")

# Check for hybrid memory-aware adjustment
if "d={self.d_features} < 15" in source or "memory-aware" in source:
    print("✓ Fix 2: Hybrid memory-aware adjustment present")
else:
    print("✗ Fix 2: Hybrid memory-aware adjustment MISSING")

print()
print("=" * 60)
print("Fix verification complete!")
print()
print("Next: Run benchmark to test fixes:")
print("  python tests/benchmarks/test_fedcdh_benchmark_v3.py \\")
print("    --datasets law_school,sachs \\")
print("    --methods fedspn_v,fedspn_hy \\")
print("    --seeds 42 \\")
print("    --device cuda \\")
print("    --save-graphs")
