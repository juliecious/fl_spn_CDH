# Smoke Tests for Causal Discovery Methods

Quick validation tests for all baseline methods and FedSPN.

## Available Tests

| Test Script | Method | Type | Device | Runtime |
|-------------|--------|------|--------|---------|
| `smoke_test_kci.py` | FedCDH-KCI | Federated (KCI) | CPU | 2-5 min |
| `smoke_test_ges.py` | GES | Centralized | CPU | < 1 min |
| `smoke_test_fci.py` | FCI | Centralized | CPU | < 1 min |
| `smoke_test_spn_h.py` | FedSPN-H | Federated (SPN) | CPU | 2-3 min |

## Configuration

All tests use:
- **Dataset**: Tiny synthetic (5 nodes, 200 samples)
- **Seed**: 42 (reproducible)
- **Device**: CPU only
- **Output**: Timestamped directories with PNG graphs

## Usage

```bash
# Run individual tests
python eval/smoke_test_kci.py
python eval/smoke_test_ges.py
python eval/smoke_test_fci.py
python eval/smoke_test_spn_h.py

# Run all tests
for test in eval/smoke_test_*.py; do python $test; done
```

## Output Structure

Each test creates a timestamped directory:
```
eval/
├── 20260514_123456_kci_5vars_200samples/
│   ├── true_graph.png          # Ground truth DAG
│   ├── discovered_graph.png    # Discovered DAG
│   ├── results.json            # Metrics (F1, precision, recall, SHD)
│   ├── summary.txt             # Human-readable summary
│   └── error.txt               # (if test failed)
```

## Success Criteria

- ✓ **PASSED**: F1 > 0.3
- ⚠ **MARGINAL**: F1 ≤ 0.3 (expected with tiny random data)
- ✗ **FAILED**: Crashes or errors

## Notes

- Small datasets may have low F1 scores due to weak signals
- These are smoke tests, not performance benchmarks
- Use full datasets (Sachs, Law School) for actual evaluation
