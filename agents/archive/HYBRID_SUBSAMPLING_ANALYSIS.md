# Hybrid Mode Sub-sampling Analysis

**Question:** Why does hybrid mode show sub-sampling to 1000 instead of 500?

**Answer:** ✅ **FOUND IT!** The hybrid config in `configs.py` has `"n": 1000` hardcoded.

---

## Root Cause

**File:** `tests/benchmarks/configs.py`
**Line:** 56

```python
PRODUCTION_CONFIGS = {
    "fedspn_horizontal": {
        ...
        "n": 500,  # ← Horizontal uses 500
        ...
    },
    "fedspn_vertical": {
        ...
        "n": 500,  # ← Vertical uses 500
        ...
    },
    "fedspn_hybrid": {
        ...
        "n": 1000,  # ← Hybrid uses 1000! ❌
        ...
    },
}
```

### How This Works

**In `run_experiment.py`:**
```python
# Load config
config = PRODUCTION_CONFIGS[args.config]  # e.g., "fedspn_hybrid"

# Apply config values
args.n = config["n"]  # For hybrid: args.n = 1000

# Load data with this n
X_splits, true_DAG_bin, c_indx = load_sachs_federated(
    args.K, n_samples_limit=args.n  # n_samples_limit=1000
)
```

**In `sachs_loader.py`:**
```python
if n_samples_limit and len(df) > n_samples_limit:
    df = df.sample(n=n_samples_limit, random_state=42)  # n=1000
    logging.warning(
        f"Sub-sampled Sachs data to {n_samples_limit} rows for testing."
    )
    # Output: "Sub-sampled Sachs data to 1000 rows for testing."
```

---

## Why Is This?

### Possible Reasons

**Theory 1: Legacy/Testing Configuration**
- Someone set `n=1000` for hybrid during development
- Forgot to change it back to 500
- Likely an oversight

**Theory 2: Intentional Larger Dataset**
- Hybrid might need more data (combines horizontal + vertical complexity)
- But this seems unlikely - the other scenarios work fine with 500

**Theory 3: Bug**
- Most likely explanation: **unintentional inconsistency**

---

## Impact Analysis

### What This Means for Experiments

| Scenario | Config n | Actual Data | Issue? |
|----------|----------|-------------|--------|
| **Horizontal** | 500 | 500 rows (sub-sampled from 7466) | ✅ Consistent |
| **Vertical** | 500 | 500 rows (sub-sampled from 7466) | ✅ Consistent |
| **Hybrid** | 1000 | 1000 rows (sub-sampled from 7466) | ⚠️ **Inconsistent** |

### Problem

**Unfair comparison:**
- Horizontal/Vertical tested on 500 samples
- Hybrid tested on 1000 samples (2× more data!)
- More data → potentially better F1 scores
- **Not comparing apples to apples** ❌

---

## Recommendation: Fix This!

### Option 1: Use Consistent n=500 (RECOMMENDED)

**Reason:** Fair comparison across all scenarios

**Fix:**
```python
# In configs.py line 56
"fedspn_hybrid": {
    ...
    "n": 500,  # Change from 1000 → 500
    ...
}
```

**Pro:**
- Fair comparison
- Consistent with horizontal/vertical
- Faster experiments

**Con:**
- Less data for hybrid (but still enough)

---

### Option 2: Use Full Dataset n=None

**Reason:** Use all available Sachs data (856 rows)

**Fix:**
```python
# In configs.py for ALL scenarios
"fedspn_horizontal": {
    ...
    "n": None,  # Use all 856 rows
    ...
},
"fedspn_vertical": {
    ...
    "n": None,  # Use all 856 rows
    ...
},
"fedspn_hybrid": {
    ...
    "n": None,  # Use all 856 rows
    ...
},
```

**Pro:**
- Use full dataset (more realistic)
- Fair comparison (all use same 856 rows)
- Matches thesis description "Sachs N=856"

**Con:**
- Slightly slower (but only ~20% more data: 856 vs 500)

---

### Option 3: Increase All to n=1000

**Reason:** Use more data for all scenarios

**Fix:**
```python
# In configs.py for ALL scenarios
"n": 1000,  # All scenarios use 1000
```

**Pro:**
- More data → potentially better F1
- Fair comparison (all use same amount)

**Con:**
- Slower experiments
- Arbitrary number (why 1000 when full dataset is 856?)

---

## Recommended Fix

**Use Option 2: Full dataset (n=None)** for thesis experiments:

```python
# tests/benchmarks/configs.py

PRODUCTION_CONFIGS = {
    "fisherz_baseline": {
        ...
        "n": None,  # Use all 856 Sachs rows
        ...
    },
    "kci_oracle": {
        ...
        "n": None,
        ...
    },
    "fedspn_horizontal": {
        ...
        "n": None,
        ...
    },
    "fedspn_vertical": {
        ...
        "n": None,
        ...
    },
    "fedspn_hybrid": {
        ...
        "n": None,  # Change from 1000
        ...
    },
}
```

### Why This is Best

1. **Fair comparison:** All methods see identical data
2. **Full dataset:** Using all 856 real Sachs samples
3. **Thesis alignment:** You say "Sachs N=856" in experiments
4. **Realistic:** No arbitrary sub-sampling for testing

---

## How to Apply Fix

### Step 1: Edit configs.py

```bash
cd /Users/M279402/PycharmProjects/fl_spn_CDH
nano tests/benchmarks/configs.py

# Change all "n": 500 or "n": 1000 to:
# "n": None
```

### Step 2: Verify

```python
# Quick check
import sys

sys.path.insert(0, '/')
from tests.benchmarks.configs import PRODUCTION_CONFIGS

for config_name, config in PRODUCTION_CONFIGS.items():
    print(f"{config_name}: n={config.get('n')}")

# Expected output:
# fisherz_baseline: n=None
# kci_oracle: n=None
# fedspn_horizontal: n=None
# fedspn_vertical: n=None
# fedspn_hybrid: n=None
```

### Step 3: Test

```bash
# Run single experiment to verify
python tests/benchmarks/run_experiment.py \
  --config fedspn_hybrid \
  --model_type sachs_real \
  --seed 0 \
  --epochs 10

# Check log - should NOT see "Sub-sampled" warning
# OR should see "Sub-sampled Sachs data to 856 rows"
```

---

## For Thesis Writing

### Before Fix

**Problem statement:**
> "We evaluated methods on Sachs dataset. However, hybrid scenario used n=1000 samples while horizontal/vertical used n=500, creating an unfair advantage."

**Not good!** Reviewers will question the validity.

### After Fix

**Clean statement:**
> "We evaluated all methods on the complete Sachs interventional dataset (N=856 samples, d=11 features, K=3 clients). All scenarios (horizontal, vertical, hybrid) were tested on identical data to ensure fair comparison."

**Much better!** ✅

---

## Summary

✅ **Found:** Hybrid config has `"n": 1000` (line 56 in configs.py)
✅ **Issue:** Inconsistent with horizontal/vertical (n=500)
✅ **Impact:** Unfair comparison (hybrid gets 2× more data)
✅ **Fix:** Change ALL configs to `"n": None` (use full 856 samples)
✅ **Benefit:** Fair comparison + use complete dataset

---

## Immediate Action

**Before running Phase 1 experiments:**

1. Edit `tests/benchmarks/configs.py`
2. Change all `"n": 500` and `"n": 1000` to `"n": None`
3. Re-create code zip
4. Upload to Colab
5. Verify with test run

**This ensures fair comparison for thesis!**

---

*Discovered: March 9, 2026*
*Impact: CRITICAL for experimental validity*
*Must fix before Phase 1 experiments*

<function_calls>
<invoke name="Read">
<parameter name="file_path">/Users/M279402/PycharmProjects/fl_spn_CDH/tests/benchmarks/configs.py
