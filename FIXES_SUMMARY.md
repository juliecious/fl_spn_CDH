# FedCDH CUDA OOM Error Handling - Fix Summary

## Problem Identified

From the evaluation results in `experiments/v3_realworld/law_school/`:

1. **FedSPN_h (horizontal)**: CUDA out of memory error during local SPN training (line 665 in FedCDH.py)
2. **FedSPN_hy (hybrid)**: All metrics returned 0.0 (likely same memory issue)
3. **FedSPN_v (vertical)**: Poor performance (skeleton_f1: 0.222)

The root cause was that SPN training code (lines 650-672 and 722-742 in FedCDH.py) had no error handling for CUDA OOM errors. When the error occurred, it would propagate up and crash the entire `fit()` method, resulting in the benchmark returning an empty graph.

## Solution Implemented

Added comprehensive error handling with automatic CPU fallback in `FedCDH.py`:

### Changes in Lines 642-729:
1. **Wrapped local cluster SPN training in try-except blocks**
   - Primary: Try training on CUDA (original device)
   - Fallback 1: On `torch.cuda.OutOfMemoryError`, clear cache and retry on CPU
   - Fallback 2: On other errors, skip the cluster with warning (allows partial training)

2. **Wrapped single SPN training in try-except blocks** (no clustering case)
   - Primary: Try training on CUDA
   - Fallback 1: On `torch.cuda.OutOfMemoryError`, clear cache and retry on CPU
   - Fallback 2: On other errors, raise RuntimeError (single SPN must succeed)

### Key Features:
- **Automatic GPU → CPU fallback**: When CUDA runs out of memory, automatically retries on CPU
- **CUDA cache clearing**: Calls `torch.cuda.empty_cache()` before CPU fallback
- **Graceful degradation**: For clustered training, allows partial success (skips failed clusters)
- **Detailed logging**: Reports which clusters/clients succeeded, failed, or used CPU fallback
- **Preserves graph return**: Even with fallback, graphs are still returned for evaluation

## Testing

Smoke test passed successfully:
```bash
$ python test_smoke.py
✓ Imports successful
✓ FedCDH imported successfully
✓ PyTorch imported (CUDA available: False)
✓ torch.cuda.OutOfMemoryError accessible: <class 'torch.cuda.OutOfMemoryError'>
✓ METHOD_REGISTRY has 6 methods
✓ DATASET_REGISTRY has 12 datasets
✓ All FedSPN methods registered
✓ law_school dataset registered
✓✓✓ ALL SMOKE TESTS PASSED ✓✓✓
```

## Next Steps

Run the full benchmark on GPU to verify the fixes work correctly:

```bash
python tests/benchmarks/test_fedcdh_benchmark_v3.py \
  --datasets law_school \
  --methods fedspn_h,fedspn_v,fedspn_hy,ges,fci \
  --seeds 42 \
  --device cuda \
  --save-graphs
```

## Expected Behavior

1. **If CUDA has sufficient memory**: Training proceeds normally on GPU
2. **If CUDA OOM occurs**: Automatically falls back to CPU with warning message
3. **If CPU also fails**:
   - For clustered training: Skips the problematic cluster, continues with others
   - For single SPN: Raises error (at least one client must succeed)

## Files Modified

- `causallearn/search/FCMBased/FedCDH/FedCDH.py`: Added error handling (lines 642-848)
- `test_smoke.py`: Created smoke test for verification

## Git Status

```bash
$ git status
On branch: v3-comprehensive-fixes
Modified:   causallearn/search/FCMBased/FedCDH/FedCDH.py
Modified:   tests/benchmarks/test_fedcdh_benchmark_v3.py (already modified)
```
