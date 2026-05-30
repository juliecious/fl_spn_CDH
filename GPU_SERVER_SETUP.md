# GPU Server Setup Instructions

## Issue
`ModuleNotFoundError: No module named 'causallearn.utils.spn.core'`

## Root Cause
The causallearn package needs to be reinstalled on the GPU server after recent structural changes to the `spn` subpackage.

## Solution

Run these commands on the GPU server:

```bash
# Navigate to project directory
cd /home/fang/fedcdh_pc

# Reinstall the package in development mode
pip install -e . --force-reinstall --no-deps

# Verify the installation
python -c "from causallearn.utils.spn.core import LocalSPNWrapper; print('✓ Import successful')"
```

## If the above doesn't work, try a clean reinstall:

```bash
# Remove existing installation
pip uninstall causal-learn -y

# Clean any cached files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

# Reinstall
pip install -e .

# Verify
python -c "from causallearn.utils.spn.core import LocalSPNWrapper; print('✓ Import successful')"
```

## Run the benchmark after reinstallation:

```bash
nohup python -m tests.benchmarks.test_fedcdh_benchmark_v3 \
  --datasets asia \
  --methods fedspn_h,fedspn_v,fedspn_hy \
  --seeds 42 \
  --K 3 \
  --device cuda \
  --output-dir eval/v3_asia_gpu_test \
  --save-graphs \
  2>&1 | tee test.log &
```

## Expected Output
```
✓ Import successful
[Benchmark] Running experiments...
```

## Troubleshooting

If imports still fail:
1. Check Python path: `echo $PYTHONPATH`
2. Check if installed: `pip show causal-learn`
3. Verify package location: `python -c "import causallearn; print(causallearn.__file__)"`
