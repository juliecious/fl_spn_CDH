#!/bin/bash
# Test Asia dataset on GPU after refactoring
# This will test all three modes: horizontal, vertical, hybrid

python -m tests.benchmarks.test_fedcdh_benchmark_v3 \
    --datasets asia \
    --methods fedspn_h fedspn_v fedspn_hy \
    --seeds 42 \
    --K 3 \
    --device cuda \
    --output-dir experiments/v3_asia_gpu_test \
    --save-graphs \
    2>&1 | tee experiments/v3_asia_gpu_test.log

echo ""
echo "========================================"
echo "Experiment completed!"
echo "Check results at: experiments/v3_asia_gpu_test/"
echo "========================================"
