#!/bin/bash
set -e

echo "=================================================="
echo "Testing All Three FedSPN Modes"
echo "=================================================="

# Test 1: Horizontal
echo -e "\n[1/3] Testing Horizontal Mode..."
sed -i '' 's/scenario=".*"/scenario="horizontal"/' smoke_test_minimal.py
sed -i '' 's/feature_maps=feature_maps_hy/feature_maps=None/' smoke_test_minimal.py
python smoke_test_minimal.py > /tmp/test_horizontal.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Horizontal PASSED"
    grep "Skeleton F1:" /tmp/test_horizontal.log | tail -1
else
    echo "✗ Horizontal FAILED"
    tail -20 /tmp/test_horizontal.log
    exit 1
fi

# Test 2: Vertical
echo -e "\n[2/3] Testing Vertical Mode..."
sed -i '' 's/scenario=".*"/scenario="vertical"/' smoke_test_minimal.py
sed -i '' 's/feature_maps=None/feature_maps=feature_maps_hy/' smoke_test_minimal.py
python smoke_test_minimal.py > /tmp/test_vertical.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Vertical PASSED"
    grep "Skeleton F1:" /tmp/test_vertical.log | tail -1
else
    echo "✗ Vertical FAILED"
    tail -20 /tmp/test_vertical.log
    exit 1
fi

# Test 3: Hybrid
echo -e "\n[3/3] Testing Hybrid Mode..."
sed -i '' 's/scenario=".*"/scenario="hybrid"/' smoke_test_minimal.py
python smoke_test_minimal.py > /tmp/test_hybrid.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Hybrid PASSED"
    grep "Skeleton F1:" /tmp/test_hybrid.log | tail -1
else
    echo "✗ Hybrid FAILED"
    tail -20 /tmp/test_hybrid.log
    exit 1
fi

echo -e "\n=================================================="
echo "✓ All Three Modes PASSED!"
echo "=================================================="
