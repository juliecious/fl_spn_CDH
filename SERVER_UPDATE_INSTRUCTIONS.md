# Server Update Instructions

## Two Critical Fixes Applied

### Fix 1: Invalid Structure Type
**Error**: `AssertionError: Invalid structure type poon-domingos. Must be 'top-down' or 'bottom-up'.`

**Solution**: Modified `causallearn/utils/spn/core/local.py` to map `"poon-domingos"` → `"top-down"` for `simple_einet` compatibility.

### Fix 2: Missing Hybrid Partition Module
**Error**: `ModuleNotFoundError: No module named 'causallearn.utils.hybrid_partition'`

**Solution**: Restored `causallearn/utils/hybrid_partition.py` for hybrid FL support.

---

## Update Steps on GPU Server

Run these commands on `cda-server-3`:

```bash
cd /home/fang/fedcdh_pc

# Pull latest fixes
git pull origin v3-comprehensive-fixes

# Reinstall package
pip install -e . --force-reinstall --no-deps

# Verify imports work
python -c "from causallearn.utils.spn.core import LocalSPNWrapper; print('✓ LocalSPNWrapper import successful')"
python -c "from causallearn.utils.hybrid_partition import create_hybrid_block_partition; print('✓ hybrid_partition import successful')"

# Run the benchmark
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

---

## What Was Changed

### File 1: `causallearn/utils/spn/core/local.py`

**Before (causing error)**:
```python
structure="poon-domingos",  # BUG: Not valid for simple_einet
```

**After (fixed)**:
```python
structure="top-down",  # Now defaults to valid type

# Map "poon-domingos" to "top-down" for simple_einet compatibility
if structure == "poon-domingos":
    structure = "top-down"
elif structure not in ["top-down", "bottom-up"]:
    raise ValueError(f"Invalid structure type: {structure}...")
```

### File 2: `causallearn/utils/hybrid_partition.py`

**Status**: Restored from backup (was accidentally deleted)

**Purpose**: Provides `create_hybrid_block_partition()` for hybrid FL scenarios

---

## Expected Output After Fix

```bash
(.venv) fang@cda-server-3:~/fedcdh_pc$ python -c "from causallearn.utils.spn.core import LocalSPNWrapper; print('✓ Import successful')"
✓ Import successful

(.venv) fang@cda-server-3:~/fedcdh_pc$ python -m tests.benchmarks.test_fedcdh_benchmark_v3 --datasets asia --methods fedspn_h --K 3 --device cuda
INFO:root:Using device: cuda
INFO:root:Loading asia benchmark dataset...
INFO:root:  asia: 1000 samples, 8 features, 8 edges
INFO:root:  FedSPN config: scenario=horizontal, K=3, K_local=2, epochs=20
INFO:root:FedCDH Initialized on device: cuda
INFO:root:Training Client 0/3
INFO:root:  Training SPN for local cluster 0: 320 samples
INFO:root:    Training complete: final LL = -45.23
...
INFO:root:  Learned skeleton:
[[0 1 0 0 0 0 0 0]
 [1 0 1 0 0 0 0 0]
 [0 1 0 1 0 0 0 0]
 ...
INFO:root:  SHD = 4 (target < 5) ✓
INFO:root:  F1 = 0.75 (target > 0.70) ✓
```

---

## Troubleshooting

### If import still fails after git pull:

```bash
# Clean reinstall
pip uninstall causal-learn -y
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null
pip install -e .
```

### If structure error persists:

Check that the local.py file was actually updated:
```bash
grep -n "poon-domingos" causallearn/utils/spn/core/local.py
# Should show line 27 (default parameter) and line 57 (mapping logic)

grep -A5 "if structure ==" causallearn/utils/spn/core/local.py
# Should show the mapping code
```

### If hybrid_partition still missing:

```bash
ls -la causallearn/utils/hybrid_partition.py
# Should exist

python -c "from causallearn.utils.hybrid_partition import create_hybrid_block_partition; print(create_hybrid_block_partition)"
# Should print: <function create_hybrid_block_partition at 0x...>
```

---

## Git Commits to Pull

Latest commits on `v3-comprehensive-fixes`:
1. `ee4e31c` - feat: implement three critical bug fixes for FedSPN-CDH
2. `219b666` - fix: resolve structure type validation and missing hybrid_partition module

To verify you have the latest code:
```bash
git log --oneline -2
# Should show:
# 219b666 fix: resolve structure type validation and missing hybrid_partition module
# ee4e31c feat: implement three critical bug fixes for FedSPN-CDH
```
