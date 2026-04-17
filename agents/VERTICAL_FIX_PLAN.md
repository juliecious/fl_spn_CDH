# Vertical/Hybrid Mode Fix Plan

**Root Cause**: Local SPNs stored for evaluation don't match the ones in FederatedProduct

---

## Issue Diagnosis

After extensive analysis, the problem is:

1. **Training**: SPNs are trained on `X_splits_train` which are extracted from `X_aug_global`
   - Client 0: features [0, 1, 2, 5] from X_aug_global (shape: 200×4)
   - Client 1: features [3, 4] from X_aug_global (shape: 200×2)

2. **Storage**: These SPNs are stored in `clients_clusters[h][k]` and then extracted to `self.local_spns`

3. **Evaluation**: When evaluating local SPNs (lines 861-893):
   - Extract features from `X_global` (NOT X_aug_global!)
   - Re-append context
   - But `X_global` at evaluation time is constructed differently than during training!

4. **Global SPN**: The FederatedProduct correctly uses the training feature_maps, so it works (as shown by my diagnostic)

---

## The Real Problem

Looking at the benchmark log again:
```
Client 0: Evaluating on features [0, 1, 2]
Train LL: -10.3408
```

The SPN was trained on **4 features** [0,1,2,context], but it's being evaluated on **3 features** [0,1,2] only!

This is happening because:
1. Line 874: `X_client = X_global[:, feature_indices_no_context]` → shape (200, 3)
2. Line 893: `X_client_aug = np.concatenate([X_client, c_client], axis=1)` → shape (200, 4)
3. Line 905: `evaluate_spn_quality(local_spn, X_client_aug, ...)` → should get (200, 4)

But somehow the SPN is seeing only 3 features!

**Hypothesis**: The issue is that for vertical mode, we need to store the TRAINING DATA alongside the SPNs so we can evaluate them correctly.

---

## Solution

### Option 1: Store Training Data (Recommended)

Store the training data splits for each client and use them for evaluation:

```python
# During training (after line 344):
self.X_splits_train = X_splits_train  # Store for evaluation

# During evaluation (replace lines 861-893):
if self.scenario == "vertical":
    # Use the ACTUAL training data for evaluation
    if hasattr(self, 'X_splits_train') and k < len(self.X_splits_train):
        X_client_aug = self.X_splits_train[k]
    else:
        # Fallback to current method
        feature_indices_no_context = self._extract_feature_indices(k, include_context=False)
        X_client = X_global[:, feature_indices_no_context]
        c_client = c_indx
        if k > 0:
            X_client_aug = X_client
        else:
            X_client_aug = np.concatenate([X_client, c_client], axis=1)
```

### Option 2: Use feature_maps Directly

Use the stored `vertical_feature_map` to extract features correctly:

```python
if self.scenario == "vertical":
    if hasattr(self, 'vertical_feature_map') and k in self.vertical_feature_map:
        indices = self.vertical_feature_map[k]
        X_client_aug = X_aug_global[:, indices]
    else:
        # Fallback...
```

---

## Recommended Fix: Option 1 + Fix Global SPN Evaluation

The global SPN evaluation also needs attention. Currently it evaluates on `X_aug_global`, but the samples it generates may not have the context column in the right place.

Let me check the sampling code...

Actually, based on my diagnostic showing FederatedProduct works correctly, I think the issue is ONLY with the local SPN evaluation, not the global one.

The global SPN shows poor LL because it's a MIXTURE of FederatedProducts (one per cluster), and if there are 2 clusters, the mixture weight might be off, or the clustering might not be good.

Let me check if the benchmark is actually using clustering...

Looking at the log:
```
BIC selection: K=5 from [...], capped to K=2 (data-driven: 200 samples)
```

So there are 2 clusters! The global SPN is a GlobalFedSPN with 2 components, each a FederatedProduct.

If one cluster has bad SPNs, the whole mixture suffers.

---

## Simplified Fix

**Just fix the local SPN evaluation to use the correct training data!**

```python
# Store training data splits
self.X_splits_train = X_splits_train  # Add after line 344

# Use them for evaluation
if self.scenario == "vertical":
    if hasattr(self, 'X_splits_train') and k < len(self.X_splits_train):
        X_client_aug = self.X_splits_train[k]
        logging.info(f"  Client {k}: Evaluating on training data (shape={X_client_aug.shape})")
```

This ensures we're evaluating on the SAME data the SPN was trained on, giving accurate LL measurements.
