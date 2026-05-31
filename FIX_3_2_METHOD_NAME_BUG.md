# FIX #3.2 Bug: Incorrect Method Name

**Date**: 2026-05-30
**Status**: ✅ Fixed

---

## Problem

Experiment `20260530_203225_asia_fedspn_h_seed42` crashed with:
```
AttributeError: 'GeneralGraph' object has no attribute 'num_edges'
```

**Location**: `causallearn/utils/PCUtils/SkeletonDiscovery.py:82`

---

## Root Cause

In FIX #3.2, I used `.num_edges` (property) instead of `.get_num_edges()` (method):

```python
# INCORRECT (Line 82)
if cg_list and len(cg_list) > 0 and cg_list[0].G.num_edges > 0:
```

The `GeneralGraph` class uses the method name `get_num_edges()`, not the property `num_edges`.

---

## Fix Applied

**File**: `causallearn/utils/PCUtils/SkeletonDiscovery.py`
**Line**: 82

```python
# CORRECT
if cg_list and len(cg_list) > 0 and cg_list[0].G.get_num_edges() > 0:
```

---

## Impact

- Previous experiments crashed before reaching the main PC algorithm
- Structure voting results were generated (22 edges found) but never used
- Pipeline terminated with error, producing default metrics (all zeros)

---

## Verification

All other uses of `.num_edges` were already checked:
- Smoke test file already used `.get_num_edges()` (was fixed earlier)
- No other instances in CDNOD.py, FedCDH.py, or SkeletonDiscovery.py

---

## Status

✅ **Fixed and ready for re-run**

Next experiment should:
1. ✅ Structure voting finds 22 edges
2. ✅ Initial skeleton passed to CDNOD
3. ✅ PC algorithm starts with 22 edges (not crash)
4. ✅ Final skeleton has > 0 edges
5. ✅ Metrics improve (SHD < 8, F1 > 0)
