# Suggested Improvements to test_fedcdh_benchmark.py

Based on the recent investigation and fixes (April 13-18, 2026).

## Current Status ✅

The main test script (`tests/test/test_fedcdh_benchmark.py`) is **production-ready** with:
- ✅ All three scenarios (H/V/Hybrid) implemented correctly
- ✅ Mixture-then-Product hybrid architecture (Week 2, April 14, 2026)
- ✅ Adaptive hyperparameters (LR, epochs, architecture scaling)
- ✅ SPN quality evaluation integrated
- ✅ Multiple configuration presets (quick/small/medium/large/sachs)

## Recommended Improvements

### 1. Add Validation Check (High Priority)

Add automated validation to detect evaluation data mismatches:

```python
# After line 200 (in run_single_scenario function)
def validate_evaluation_consistency(fedcdh):
    """Verify evaluation uses stored training data."""
    if hasattr(fedcdh, 'X_aug_global_train'):
        logging.info("✓ Evaluation fix verified: X_aug_global_train stored")
        return True
    else:
        logging.warning("⚠️  Evaluation may use reconstructed data")
        return False

# Call after fedcdh.fit()
validate_evaluation_consistency(fedcdh)
```

**Why**: Ensures the hybrid/vertical fix is working in future runs.

---

### 2. Add Expected LL Ranges (Medium Priority)

Add sanity checks for SPN log-likelihood values:

```python
# After evaluation results are logged
def check_ll_sanity(scenario, d, local_lls, global_ll):
    """Warn if LL values are suspiciously poor."""
    # Expected ranges based on investigation
    if scenario == "hybrid":
        expected_global = (-8, -15)  # After fix
        if global_ll < expected_global[1]:
            logging.warning(
                f"⚠️  Hybrid global LL ({global_ll:.2f}) unexpectedly poor. "
                f"Expected range: {expected_global}. Check evaluation fix."
            )
    elif scenario == "vertical":
        expected_global = (-5, -15)  # After fix
        if global_ll < expected_global[1]:
            logging.warning(
                f"⚠️  Vertical global LL ({global_ll:.2f}) unexpectedly poor. "
                f"Expected range: {expected_global}. Check evaluation fix."
            )
```

**Why**: Early detection of evaluation issues before full analysis.

---

### 3. Add Quick Validation Mode (Medium Priority)

Add a `--validate` flag that runs fast sanity checks:

```python
if args.validate:
    logging.info("Running validation mode (quick checks only)...")

    # Test 1: Compliance check
    from tests.validation.verify_federated_compliance import verify_compliance
    verify_compliance()

    # Test 2: Fix verification
    from tests.validation.verify_hybrid_fix import verify_fix
    verify_fix()

    # Test 3: Quick smoke test (d=5, K=2, n=200, 10 epochs)
    run_single_scenario(config="quick", scenario="hybrid", ...)

    logging.info("✅ Validation passed!")
    sys.exit(0)
```

**Usage**: `python tests/test/test_fedcdh_benchmark.py --validate`

**Why**: Fast pre-commit verification (~2 minutes vs 1+ hour full benchmark).

---

### 4. Improve Results Logging (Low Priority)

Add structured results output:

```python
# After each scenario completes
results_dict = {
    'timestamp': timestamp,
    'scenario': scenario,
    'config': config_name,
    'local_lls': local_lls,
    'global_ll': global_ll,
    'skeleton_f1': skeleton_f1,
    'overall_f1': overall_f1,
    'runtime_secs': runtime,
    'evaluation_fix_applied': hasattr(fedcdh, 'X_aug_global_train'),
}

# Save to JSON
import json
results_file = f"{output_dir}/results_{scenario}_{seed}.json"
with open(results_file, 'w') as f:
    json.dump(results_dict, f, indent=2)
```

**Why**: Easier programmatic analysis of multiple runs.

---

### 5. Add Comparison Mode (Low Priority)

Add flag to compare before/after fix results:

```python
parser.add_argument(
    '--compare-baseline',
    type=str,
    help='Path to baseline results JSON for comparison'
)

if args.compare_baseline:
    baseline = json.load(open(args.compare_baseline))
    current = results_dict

    improvement = current['global_ll'] - baseline['global_ll']
    logging.info(f"Improvement over baseline: {improvement:.2f}")

    if scenario == 'hybrid' and improvement < 5:
        logging.warning("Expected ~2× improvement not seen!")
```

**Why**: Quantify impact of fixes in future work.

---

## Priority Implementation Order

1. **Validation Check** (5 minutes) - Add after line 200
2. **Expected LL Ranges** (10 minutes) - Add sanity checks
3. **Quick Validation Mode** (30 minutes) - New CLI flag
4. **Results Logging** (15 minutes) - JSON output
5. **Comparison Mode** (20 minutes) - Baseline comparison

**Total Time**: ~1.5 hours to implement all improvements

---

## Current Test Coverage ✅

The existing test script already covers:
- ✅ All three scenarios (H/V/Hybrid)
- ✅ Multiple data types (linear/nonlinear)
- ✅ Multiple configurations (quick → large)
- ✅ Multiple seeds for statistical significance
- ✅ SPN quality evaluation (LL, MMD, KS tests)
- ✅ Independence structure evaluation
- ✅ UMAP visualizations
- ✅ Comprehensive logging

**Verdict**: Script is production-ready. Suggested improvements are **optional enhancements** for future robustness.

---

## Breaking Changes: None

All suggestions are **additive** - no breaking changes to existing functionality.

---

## Alternative: Keep As-Is ✅

The current test script is **sufficient for thesis**. These improvements are nice-to-have but not required.

**Recommendation**: Implement #1 (Validation Check) only for peace of mind. Rest are optional.
