"""
Debug script to diagnose vertical mode log-likelihood issue.
"""
import numpy as np
import torch
import sys
import os

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from causallearn.utils.FedPC import LocalSPNWrapper, FederatedProduct

# Simulate the vertical mode setup
np.random.seed(42)
torch.manual_seed(42)

d = 5
n = 200

# Generate synthetic data
X_data = np.random.randn(n, d)  # 5 features
context = np.random.randint(0, 2, size=(n, 1))  # Binary context
X_aug = np.concatenate([X_data, context], axis=1)  # Shape: (200, 6)

print("=" * 60)
print("VERTICAL MODE DEBUG")
print("=" * 60)
print(f"Data shape: X_data={X_data.shape}, X_aug={X_aug.shape}")
print(f"Features: 0-4, Context: 5")
print()

# Vertical split (simulating FedCDH lines 336-344)
K = 2
cols_per_client = np.array_split(range(d), K)
feature_maps = {}
X_splits_train = []

for k in range(K):
    f_indices = cols_per_client[k].tolist()
    if k == 0:
        f_indices.append(d)  # Add context column
    feature_maps[k] = f_indices
    X_splits_train.append(X_aug[:, f_indices])
    print(
        f"Client {k}: feature_indices={f_indices}, train_data_shape={X_splits_train[k].shape}"
    )

print()

# Train local SPNs
local_spns = []
for k in range(K):
    X_train_k = X_splits_train[k]
    local_d = X_train_k.shape[1]

    spn_k = LocalSPNWrapper(
        num_features=local_d,
        device="cpu",
        num_sums=20,
        num_leaves=20,
        depth=int(np.floor(np.log2(local_d))),
        num_repetitions=10,
        seed=k,
    )

    final_ll = spn_k.train_local(X_train_k, epochs=20, lr=0.01)
    local_spns.append(spn_k)

    # Compute train LL
    X_torch = torch.tensor(X_train_k, dtype=torch.float32)
    with torch.no_grad():
        train_ll = spn_k.log_prob(X_torch).mean().item()

    print(f"Client {k}: trained on {local_d} features, train_ll={train_ll:.4f}")

print()

# Create FederatedProduct
fed_product = FederatedProduct(
    clients=local_spns, feature_map=feature_maps, device="cpu"
)

print("=" * 60)
print("EVALUATION")
print("=" * 60)

# Test 1: Evaluate on training data (correct setup)
print("\nTest 1: Evaluate on X_aug (full data with context)")
X_aug_torch = torch.tensor(X_aug, dtype=torch.float32)
with torch.no_grad():
    global_ll_aug = fed_product.log_prob(X_aug_torch).mean().item()
print(f"Global LL (X_aug): {global_ll_aug:.4f}")

# Test 2: Manually compute expected LL (sum of local LLs)
print("\nTest 2: Manual computation (sum of local LLs)")
with torch.no_grad():
    ll_0 = (
        local_spns[0]
        .log_prob(torch.tensor(X_splits_train[0], dtype=torch.float32))
        .mean()
        .item()
    )
    ll_1 = (
        local_spns[1]
        .log_prob(torch.tensor(X_splits_train[1], dtype=torch.float32))
        .mean()
        .item()
    )
print(f"Client 0 LL: {ll_0:.4f}")
print(f"Client 1 LL: {ll_1:.4f}")
print(f"Sum: {ll_0 + ll_1:.4f}")

# Test 3: Check what FederatedProduct extracts
print("\nTest 3: Check feature extraction")
for k in range(K):
    indices = feature_maps[k]
    x_local = X_aug[:, indices]
    print(f"Client {k}: extracts X_aug[:, {indices}], shape={x_local.shape}")

    # Check if this matches training data
    matches = np.allclose(x_local, X_splits_train[k])
    print(f"  Matches training data: {matches}")

    if not matches:
        print(f"  Max difference: {np.abs(x_local - X_splits_train[k]).max()}")

# Test 4: Evaluate without context column (WRONG but let's see)
print("\nTest 4: Evaluate on X_data (NO context) - SHOULD FAIL")
X_data_torch = torch.tensor(X_data, dtype=torch.float32)
try:
    with torch.no_grad():
        global_ll_no_context = fed_product.log_prob(X_data_torch).mean().item()
    print(f"Global LL (no context): {global_ll_no_context:.4f}")
except Exception as e:
    print(f"ERROR (expected): {e}")

print()
print("=" * 60)
print("DIAGNOSIS")
print("=" * 60)

print(
    """
Expected behavior:
- Local Client 0 LL: ~-5 to -6 (3 features)
- Local Client 1 LL: ~-3 to -4 (2 features)
- Global LL: ~-8 to -10 (sum of locals)

If Global LL is much worse (-16 to -17), possible causes:
1. Normalization mismatch (mean/std computed wrong)
2. Feature extraction bug (wrong columns)
3. Context column handling issue
4. Training didn't converge properly
"""
)

# Additional diagnostic: Check normalization stats
print("\nNormalization stats check:")
for k in range(K):
    spn = local_spns[k]
    if spn.mean is not None:
        print(f"Client {k}:")
        print(f"  mean: {spn.mean.cpu().numpy()}")
        print(f"  std: {spn.std.cpu().numpy()}")
