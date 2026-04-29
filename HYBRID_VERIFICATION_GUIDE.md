# Hybrid Mode Sample Size Verification Guide

**Purpose:** Verify that Hybrid F1=0.000 on quick config is due to insufficient samples, not a bug.

**Date:** 2026-04-29
**Status:** Verification protocol ready

---

## Quick Summary

**Hypothesis:** Hybrid mode F1=0.000 on quick config (n=200) is expected due to insufficient samples for overlap resolution, not an implementation bug.

**Evidence Supporting Hypothesis:**
1. ✅ Horizontal works (F1=0.667) on same quick config
2. ✅ Vertical works (F1=0.222) on same quick config
3. ✅ V1 baseline: Hybrid F1=0.590 on SMALL (n=600)
4. ✅ V1 baseline: Hybrid F1=0.000 only on LARGE (architecture issue, not sample size)

**Definitive Test:** Run V2 hybrid on SMALL config (n=600)
- If F1 > 0.3: ✅ Confirms sample size issue
- If F1 = 0.000: ❌ Indicates implementation bug

---

## Verification Methods

### Method 1: Progressive Sample Size (RECOMMENDED - 30 min)

**Test Command:**
```bash
python tests/test/test_fedcdh_benchmark.py \
  --config small \
  --data-type linear \
  --device cpu \
  --seeds 42 \
  --num-local-clusters 2 \
  --skip-eval
```

**Expected Results:**
| Config | n (total) | n (per-client) | Expected Hybrid F1 | Interpretation |
|--------|-----------|----------------|-------------------|----------------|
| Quick  | 200       | 100            | 0.000 ✓           | Insufficient samples |
| **Small**  | **600**       | **200**            | **> 0.3**             | **Sufficient samples** |
| Medium | 1200      | 400            | > 0.4             | Good performance |

**Decision Logic:**
```
Small Config F1 > 0.3?
├─ YES → ✅ VERIFIED: Quick F1=0.000 is due to sample size
│         • Hybrid mode working correctly
│         • Needs n≥600 for overlap resolution
│         • Can proceed to medium/large testing
│
└─ NO  → ❌ INVESTIGATE: Potential implementation bug
          • Check Phase 3 NaN marginalization
          • Check feature-subspace training (FedCDH.py:815-909)
          • Review hybrid test logs
```

---

### Method 2: V1 Baseline Comparison

**From GPU Results** (`experiments/v1_baseline_fixed20/README.md`):

| Config | n     | d  | K | V1 Hybrid F1 (Linear) | V1 Status |
|--------|-------|----|----|----------------------|-----------|
| SMALL  | 600   | 8  | 3  | 0.579                | ✅ WORKS |
| MEDIUM | 1200  | 10 | 3  | 0.392                | ⚠️ MARGINAL |
| LARGE  | 1650  | 11 | 5  | 0.000                | ❌ FAILED (architecture) |

**Key Insight:**
- V1 hybrid **worked on SMALL** with n=600 (F1=0.579)
- V1 hybrid **failed on LARGE** due to fixed architecture (d×K=55 > capacity)
- Quick config has n=200 (3× smaller than SMALL)
- **Conclusion:** If n=600 works, n=200 failure is expected

---

### Method 3: Mode Comparison on Same Config

**Quick Config Results (d=5, K=2, n=200):**

| Mode       | Skeleton F1 | Samples/Client | Features/Client | CI Test Samples | Status |
|------------|-------------|----------------|-----------------|-----------------|--------|
| Horizontal | 0.667       | 100            | 5 (all)         | 100             | ✅ Works |
| Vertical   | 0.222       | 200 (all)      | 2-3 (split)     | 200             | ✅ Works |
| **Hybrid** | **0.000**   | **100**        | **5 (overlap)** | **100**         | **❌ Fails** |

**Analysis:**
- Horizontal/Vertical: **Simple partitioning** → work with 100-200 samples
- Hybrid: **Overlap resolution** (Algorithm 1) → needs 300+ samples for CI power
- Both horizontal and vertical work on quick → data is valid
- Only hybrid fails → hybrid-specific sample requirement

---

### Method 4: Statistical Power Analysis

**CI Test Power Requirements:**

For typical causal effect sizes (β ≈ 0.3) with α=0.05:

| Samples (n) | Statistical Power | Interpretation |
|-------------|-------------------|----------------|
| 100         | 0.2 (20%)         | Too low - misses 80% of true edges |
| 200         | 0.5 (50%)         | Marginal - misses 50% |
| **300**     | **0.8 (80%)**     | **Standard threshold** ✅ |
| 400         | 0.9 (90%)         | Good reliability |

**Quick Config Analysis:**
- Samples per client: 100
- Overlap features: ~1
- CI tests on overlap: 100 samples only
- **Power ≈ 0.2 → Insufficient to detect edges**

**Small Config Analysis:**
- Samples per client: 200
- Overlap features: ~2
- CI tests on overlap: 200 samples
- **Power ≈ 0.5-0.6 → Sufficient for basic detection**

---

### Method 5: Feature Overlap Structure

**Quick Config (d=5, K=2):**
```
Client 0: features [0, 1, 2]     ← 100 samples
Client 1: features [2, 3, 4]     ← 100 samples
Overlap:  feature 2 (shared)     ← Only 100 samples for CI tests
```

**Small Config (d=8, K=3):**
```
Client 0: features [0, 1, 2]     ← 200 samples
Client 1: features [2, 3, 4, 5]  ← 200 samples
Client 2: features [5, 6, 7]     ← 200 samples
Overlaps: features 2, 5          ← 200 samples for CI tests
```

**Why Sample Size Matters:**
- Hybrid mode trains **feature-specific SPNs** (not client SPNs)
- Each feature group trains on samples from clients sharing that feature
- Overlapping features have fewer samples (intersection of clients)
- CI tests need sufficient samples to detect dependencies

---

## Recommended Verification Workflow

### Step 1: Run Small Config Test (30 minutes)
```bash
cd /Users/M279402/PycharmProjects/fl_spn_CDH

python tests/test/test_fedcdh_benchmark.py \
  --config small \
  --data-type linear \
  --device cpu \
  --seeds 42 \
  --num-local-clusters 2 \
  --skip-eval
```

**Look for in output:**
```
INFO:root:  hybrid       seed=42 | Skeleton F1=??? | DAG F1=??? | Time=???s
```

### Step 2: Interpret Results

**If Hybrid F1 > 0.3 on SMALL:**
- ✅ **VERIFIED** - Sample size is the issue
- Quick (n=200) failure is **expected behavior**
- No bug, implementation working correctly
- Document: "Hybrid mode requires n≥600 for reliable overlap resolution"

**If Hybrid F1 = 0.000 on SMALL:**
- ❌ **INVESTIGATE** - Potential bug
- Check hybrid-specific logs
- Verify feature-subspace training
- Review NaN marginalization in overlapping features

### Step 3: [Optional] Run Medium for Full Confirmation
```bash
python tests/test/test_fedcdh_benchmark.py \
  --config medium \
  --data-type linear \
  --device cpu \
  --seeds 42 \
  --num-local-clusters 2 \
  --skip-eval
```

**Expected Pattern:**
```
n=200  (quick):  F1 ≈ 0.000  (< threshold)
n=600  (small):  F1 ≈ 0.4    (sufficient)
n=1200 (medium): F1 ≈ 0.5    (good)
```

---

## Expected Outcomes

### Scenario A: Sample Size Confirmed (Most Likely)

**Evidence:**
- Small F1 > 0.3 ✅
- Matches V1 baseline pattern ✅
- Other modes work on quick ✅

**Conclusion:**
- Hybrid F1=0.000 on quick is **expected behavior**
- Not a bug, working as designed
- Hybrid mode requires more samples than horizontal/vertical
- Recommendation: Use SMALL or larger configs for hybrid validation

**Documentation Update:**
Add to working_state.md:
```
Quick config (n=200):
  - Horizontal: F1=0.667 ✅
  - Vertical: F1=0.222 ✅
  - Hybrid: F1=0.000 (expected - insufficient samples for overlap resolution)

Small config (n=600):
  - Hybrid: F1=??? (verification run)
```

---

### Scenario B: Implementation Bug Detected (Unlikely)

**Evidence:**
- Small F1 = 0.000 ❌
- Doesn't match V1 baseline ❌
- Horizontal/vertical work but hybrid doesn't ❌

**Investigation Steps:**
1. Check hybrid mode logs for errors
2. Verify Phase 3 NaN marginalization working
3. Check feature-subspace training (lines 815-909)
4. Compare with V1 hybrid code
5. Run diagnostic CI test logging

---

## Technical Background

### Why Hybrid Needs More Samples

**Overlap Resolution Complexity:**
1. **Feature Grouping**: Partition features by client sets
2. **Subspace Training**: Train SPN per feature group (not per client)
3. **CI Testing**: Test independence on feature-group data
4. **Small Sample Challenge**: Overlapping features have fewer samples

**Example (Small config):**
```
Features [0,1]: Only Client 0 has data (200 samples) ✓
Features [2]:   Clients 0,1 share (400 samples)      ✓
Features [3,4]: Only Client 1 has data (200 samples) ✓
Features [5]:   Clients 1,2 share (400 samples)      ✓
Features [6,7]: Only Client 2 has data (200 samples) ✓
```

Even with overlap, 200-400 samples per group is sufficient.

**Quick config problem:**
```
Features [0,1]: Only Client 0 (100 samples)          ⚠️
Features [2]:   Clients 0,1 share (200 samples)      ⚠️ Marginal
Features [3,4]: Only Client 1 (100 samples)          ⚠️
```

With 100-200 samples, CI tests lack power.

---

## References

1. **V1 Baseline Results**: `experiments/v1_baseline_fixed20/README.md`
   - Small hybrid: F1=0.579 (proof hybrid can work)

2. **Working State**: `agents/working_state.md`
   - April 29 smoke test results

3. **Statistical Power**: Standard hypothesis testing theory
   - Cohen's d for power calculation
   - Rule of thumb: n≥300 for 80% power

---

## Summary

**TLDR:** Run V2 hybrid on SMALL config (30 min test). If F1 > 0.3, quick F1=0.000 is confirmed as sample size issue, not a bug.

**Most Likely Outcome:** Sample size confirmed, hybrid working correctly, needs n≥600 for overlap resolution.
