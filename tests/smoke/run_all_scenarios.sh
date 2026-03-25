#!/bin/bash
# Test all three FedCDH scenarios: Horizontal, Vertical, Hybrid
# Runtime: ~20-30 seconds on M1 Mac
# Date: 2026-03-06

set -e

# Get to repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "========================================="
echo "All Scenarios Test (Improved)"
echo "Testing: Horizontal, Vertical, Hybrid"
echo "Dataset: 6 nodes, 500 samples, 2 clients, 20 epochs"
echo "Date: $(date)"
echo "========================================="
echo ""

PYTHONPATH="$REPO_ROOT:$PYTHONPATH" python tests/smoke/test_all_scenarios.py

echo ""
echo "========================================="
echo "✓ All scenarios test complete!"
echo "Results saved to: tests/results/"
echo "========================================="
