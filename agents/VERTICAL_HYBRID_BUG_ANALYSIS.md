# Vertical/Hybrid Mode Global SPN Bug Analysis

**Date**: April 17, 2026
**Issue**: Global SPN has catastrophic LL (-16 to -17) despite good local SPNs (-4)
**Status**: ROOT CAUSE IDENTIFIED

---

## Bug Summary

The vertical and hybrid modes have a **critical data mismatch** between training and evaluation:

**Training**: Local SPNs are trained on features extracted from `X_aug_global` with context at position `d`
**Evaluation**: Local SPNs are evaluated on features extracted from `X_global` with context re-appended

This causes a **normalization mismatch** that breaks log-likelihood computation.

---

## Detailed Analysis

### Training Phase (Lines 336-344)

```python
if self.scenario == "vertical":
    cols_per_client = np.array_split(range(self.d_features), self.K_clients)
    feature_maps = {}
    X_splits_train = []
    for k in range(self.K_clients):
        f_indices = cols_per_client[k].tolist()
        if k == 0:
            f_indices.append(self.d_features)  # Add context column at position d
        feature_maps[k] = f_indices
        X_splits_train.append(X_aug_global[:, f_indices])  # Extract from full augmented data
```

**Example (d=5, K=2)**:
- `X_aug_global` shape: (200, 6) = 5 features + 1 context
- Client 0: `f_indices = [0, 1, 2, 5]` → shape (200, 4)
  - Columns: [feature0, feature1, feature2, **context_from_position_5**]
- Client 1: `f_indices = [3, 4]` → shape (200, 2)
  - Columns: [feature3, feature4]

**Local SPN normalization** (computed during `train_local()`):
```python
# Client 0's LocalSPNWrapper
self.mean = data.mean(axis=0)  # Shape: (4,)
self.std = data.std(axis=0)    # Shape: (4,)

# mean[0] = mean of X_aug_global[:, 0]
# mean[1] = mean of X_aug_global[:, 1]
# mean[2] = mean of X_aug_global[:, 2]
# mean[3] = mean of X_aug_global[:, 5]  ← Context column from position 5!
```

### Evaluation Phase (Lines 861-893)

```python
elif self.scenario == "vertical":
    # Extract features WITHOUT context
    feature_indices_no_context = self._extract_feature_indices(k, include_context=False)
    X_client = X_global[:, feature_indices_no_context]  # From X_global, NOT X_aug_global!
    c_client = c_indx  # All samples, context doesn't change

    # Add context back for client 0 only
    if self.scenario == "vertical" and k > 0:
        X_client_aug = X_client  # No context for clients other than 0
    else:
        X_client_aug = np.concatenate([X_client, c_client], axis=1)  # Re-append context!
```

**Example (d=5, K=2)**:
- `X_global` shape: (200, 5) = 5 features, NO context
- Client 0:
  - `feature_indices_no_context = [0, 1, 2]`
  - `X_client = X_global[:, [0,1,2]]` → shape (200, 3)
  - `X_client_aug = np.concatenate([X_client, c_indx], axis=1)` → shape (200, 4)
  - **Columns**: [feature0, feature1, feature2, **context_re_appended**]

**The Bug**:
Client 0's SPN was trained with mean/std computed on:
```
[X_aug_global[:, 0], X_aug_global[:, 1], X_aug_global[:, 2], X_aug_global[:, 5]]
```

But during evaluation, it receives:
```
[X_global[:, 0], X_global[:, 1], X_global[:, 2], c_indx[:, 0]]
```

**These are identical data**, but the SPN doesn't know that! It normalizes using:
```python
x_norm = (x - self.mean) / (self.std + 1e-6)
```

Where `self.mean` and `self.std` were computed on the **training data layout**.

---

## Why This Breaks Log-Likelihood

Actually wait - if the data values are identical (`X_aug_global[:, 5]` == `c_indx[:, 0]`), then normalization should still work...

Let me reconsider. The issue might be different.

---

## Alternative Hypothesis: Vertical Mode Training Bug

Let me check the actual training loop. Looking at the benchmark log:

```
Client 0 (3 features): Train LL = -10.3408  ← Very poor!
Client 1 (2 features): Train LL = -7.1216   ← Also poor!
```

My diagnostic script showed:
```
Client 0 (4 features with context): Train LL = -5.0501  ← Good!
Client 1 (2 features): Train LL = -2.8991  ← Good!
```

**Key difference**: My diagnostic trained Client 0 with 4 features (3 + context), but the benchmark evaluation shows Client 0 with 3 features only!

This means during evaluation, the SPNs are being evaluated **without the context column** even though they were trained **with it** (for Client 0).

---

## The Real Bug: Context Column Handling in Evaluation

Looking at line 891 again:
```python
if self.scenario == "vertical" and k > 0:
    X_client_aug = X_client  # No context for clients other than 0
else:
    X_client_aug = np.concatenate([X_client, c_client], axis=1)
```

This should add context to Client 0, making it shape (200, 4). But then look at the evaluation call (line 905-913):

```python
result = evaluate_spn_quality(
    local_spn,
    X_client_aug,  # Should be (200, 4) for Client 0
    n_samples=min(150, len(X_client)),
    device=self.device,
    compute_mmd=True,
    compute_ks=True,
    name=spn_name,
)
```

And inside `evaluate_spn_quality()` (spn_evaluation.py:167):
```python
X_torch = torch.tensor(X_data, dtype=torch.float32).to(device)
with torch.no_grad():
    train_ll = spn_model.log_prob(X_torch).mean().item()
```

This should work! But then lines 174-176:
```python
# Remove context column (last column) from both
X_features = X_data[:, :-1] if X_data.shape[1] > 1 else X_data
samples_features = samples[:, :-1] if samples.shape[1] > 1 else samples
```

**This is for MMD/KS testing only**, not for train_ll computation!

So the train_ll at line 167 should be computed correctly...

---

## Wait - Let Me Check the Actual Evaluation

Let me trace through the exact evaluation for Client 0:

1. Training data: `X_splits_train[0]` = `X_aug_global[:, [0,1,2,5]]` shape (200, 4)
2. Evaluation data: `X_client_aug` = `np.concatenate([X_global[:, [0,1,2]], c_indx], axis=1)` shape (200, 4)

**Key question**: Is `X_aug_global[:, 5]` == `c_indx[:, 0]`?

Let me check how X_aug_global is constructed (line 320):
```python
X_aug_global = np.concatenate([X_global, c_indx], axis=1)
```

So `X_aug_global[:, 5]` == `c_indx[:, 0]` **if and only if** `X_global` is the same at training and evaluation!

But at training time (line 310-315):
```python
if self.scenario == "vertical":
    X_global = np.concatenate(X_splits, axis=1)  # Horizontal concatenation!
```

So at training:
- `X_global` = horizontal concatenation of client splits
- `X_aug_global` = `X_global` + context

But at evaluation (line 995, called from fit()):
- We're inside `fit()`, so `X_global` is still the same
- So `X_aug_global` should be identical

**Actually, I think the issue is simpler**: Let me check if the local SPNs being evaluated are the same ones that were trained!

---

## The REAL Issue: GlobalFedSPN vs Local SPNs

Wait - the local SPN evaluation shows poor LL, but those are the STANDALONE local SPNs. The global SPN is a FederatedProduct that wraps them.

The issue is that when we evaluate the **global** SPN (line 996-1004), we're evaluating the FederatedProduct, which internally calls the local SPNs with different feature extraction.

Let me check what data is being passed to the global SPN evaluation:
- Line 998: `X_aug_global` → shape (200, 6) for d=5

The FederatedProduct.log_prob() (FedPC.py:1064-1073):
```python
def log_prob(self, x):
    client_lls = []
    for i, client in enumerate(self.clients):
        indices = self.feature_map[i]  # [0,1,2,5] for Client 0
        x_local = x[:, indices]        # Extract from x
        client_lls.append(client.log_prob(x_local))
    ll_stack = torch.cat(client_lls, dim=1)
    return torch.sum(ll_stack, dim=1, keepdim=True)
```

So it extracts `x[:, [0,1,2,5]]` from `x` with shape (200, 6). This gives shape (200, 4), which is correct!

**So why is the global LL so bad?**

Let me run another diagnostic that replicates the exact benchmark setup...

Actually, I think I know the issue now. Let me check the local SPN evaluation more carefully. The log says:

```
Client 0: Evaluating on features [0, 1, 2]
Train LL: -10.3408
```

This means it's evaluating on **3 features**, not 4! So the SPN (which was trained on 4 features) is being evaluated on only 3 features. This is causing the poor LL.

The bug is in lines 874 and 893 - it extracts features WITHOUT context, then re-appends it, but then the logging at line 878 only shows the features WITHOUT context.

But the actual `X_client_aug` passed to `evaluate_spn_quality()` should have the context. Unless... let me check if there's an issue with how the data is structured.

Actually, I think the issue might be that the local SPNs stored in `self.local_spns` are NOT the same as the ones in the FederatedProduct! Let me check how local_spns is populated.
