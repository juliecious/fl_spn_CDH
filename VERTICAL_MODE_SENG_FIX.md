# Vertical Mode: Seng's Mixture-of-Products Architecture

## Problem: Why Old Vertical Mode Failed

**Old Architecture (WRONG):**
```
ProductOverGroups:
  └─ Product: [client1_mixture] × [client2_mixture] × [client3_mixture]
```

**Formula:**
```
P(X1, X2, X3) = P(X1) × P(X2) × P(X3)
```

**Issue:** Forces complete independence between clients' features!
- For Law School: race (client 0) and ZFYA (client 2) are treated as independent
- Result: P-values = 1.000 for all cross-client edges
- Only edges within same client can be detected

## Solution: Seng's Mixture-of-Products

**New Architecture (Seng et al. 2025):**
```
GlobalSumOfProducts (ROOT):
  ├─ Product_1 (L=1): [client1_cluster_0] × [client2_cluster_0] × [client3_cluster_0]
  ├─ Product_2 (L=2): [client1_cluster_0] × [client2_cluster_0] × [client3_cluster_1]
  ├─ Product_3 (L=3): [client1_cluster_0] × [client2_cluster_1] × [client3_cluster_0]
  ├─ Product_4 (L=4): [client1_cluster_0] × [client2_cluster_1] × [client3_cluster_1]
  ├─ Product_5 (L=5): [client1_cluster_1] × [client2_cluster_0] × [client3_cluster_0]
  ├─ Product_6 (L=6): [client1_cluster_1] × [client2_cluster_0] × [client3_cluster_1]
  ├─ Product_7 (L=7): [client1_cluster_1] × [client2_cluster_1] × [client3_cluster_0]
  └─ Product_8 (L=8): [client1_cluster_1] × [client2_cluster_1] × [client3_cluster_1]
```

**Formula (Seng's Assumption 2):**
```
P(X1, X2, X3) = Σ_l q(L=l) × Π_i p(Xi | L=l)
```

**Key Insight:**
- **Latent variable L** (cluster combinations) captures cross-client dependencies
- Within each cluster combination (L=l), features are conditionally independent
- But marginally (over all L), features CAN be dependent!

## Example: How It Captures Dependencies

**Law School: race (X1, client 0) → ZFYA (X3, client 2)**

Suppose we learn:
- **Product_1 (L=1)**: minority race cluster × high ZFYA cluster → weight = 0.3
- **Product_8 (L=8)**: majority race cluster × low ZFYA cluster → weight = 0.4
- Other products: lower weights

Then:
```
P(race=minority, ZFYA=high) ≈ 0.3 × p1(race=minority) × p3(ZFYA=high)
P(race=majority, ZFYA=low) ≈ 0.4 × p8(race=majority) × p3(ZFYA=low)
```

The **correlation** is captured by the mixture weights!
- High L=1 weight → "minority race often co-occurs with high ZFYA"
- High L=8 weight → "majority race often co-occurs with low ZFYA"

## Implementation Details

### Step 1: Cluster Combinations Sampling

```python
combinations, combo_weights = sample_cluster_combinations(
    K_clients=3,
    K_local=2,
    num_samples=None,  # Enumerate all if K_local^K <= 20
    seed=42,
)
# Returns: 8 combinations for K=3, K_local=2
# [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]
```

**Strategy:**
- If K_local^K ≤ 20: Enumerate all combinations (exact)
- If K_local^K > 20: Random sample 10-20 combinations (approximate)

### Step 2: Build Products

For each cluster combination `cluster_config = [c1, c2, c3]`:

```python
for k in range(K_clients):
    cluster_idx = cluster_config[k]  # Which cluster for client k
    cluster_spn = client_local_mixtures[k].cluster_spns[cluster_idx]

    # Wrap cluster SPN for this feature subset
    group_mix = GroupMixture(
        client_spns=[cluster_spn],
        weights=[1.0],
        feature_indices=feature_maps[k],  # e.g., [0, 1] for client 0
        device=self.device,
    )
    group_mixtures.append(group_mix)

# Create product: p(X | L=l) = Π_i p(Xi | L=l)
product = ProductOverGroups(
    group_mixtures=group_mixtures,
    feature_groups=feature_groups,
    device=self.device,
)
```

### Step 3: Build Mixture

```python
fed_spn = GlobalSumOfProducts(
    products=products,  # All P products
    weights=combo_weights,  # Prior q(L=l)
    device=self.device,
)
```

## Complexity Analysis

**Number of Products (P):**
- P = K_local^K (all combinations)
- Examples:
  - K=3, K_local=2 → P=8 ✓ manageable
  - K=4, K_local=2 → P=16 ✓ manageable
  - K=5, K_local=3 → P=243 ✗ too many → sample 10-20

**Memory:**
- Old: 1 product × K clients = K models
- New: P products × K clients = P×K models
- But: Models are shared (reuse cluster SPNs)
- Actual memory: K × K_local cluster SPNs (same as before!)

**Inference Cost:**
- Old: O(K) forward passes
- New: O(P × K) forward passes
- For K=3, K_local=2: 8×3=24 vs 3 → 8× slower
- But: Enables cross-client dependency learning!

## Expected Performance Improvement

### Law School (d=5, K=3)

**Before (old vertical):**
```
Ground truth edges: 7
Detected edges: 2 (only race ⊥̸ LSAT on same client)
Missed edges: 5 (all cross-client)
F1: 0.222
P-values for cross-client: 1.000 (forced independence)
```

**After (Seng's vertical):**
```
Expected detected edges: 5-6
Expected F1: 0.60-0.75
P-values for cross-client: 0.01-0.10 (now detectable!)
```

### Why It Works

**Cross-client edge: race (client 0) → ZFYA (client 2)**

**Old CI test:**
```
ll(race, ZFYA) = ll(race) + ll(ZFYA)  ← forced independence
score = 0.000
p-value = 1.000
```

**New CI test:**
```
ll(race, ZFYA) = log[Σ_l q(L=l) × p(race|L=l) × p(ZFYA|L=l)]
                ≠ log[Σ_l q(L=l) × p(race|L=l)] + log[Σ_l q(L=l) × p(ZFYA|L=l)]
score > 0 if race and ZFYA cluster together
p-value < 0.05
```

## Code Changes Summary

**File:** `causallearn/search/FCMBased/FedCDH/FedCDH.py`

**Lines:** ~1168-1209 (vertical mode aggregation)

**Changes:**
1. Import `GlobalSumOfProducts` and `sample_cluster_combinations`
2. Generate cluster combinations (Algorithm 1, line 13)
3. For each combination, build a product (lines 14-18)
4. Build mixture over products (line 19)

**Key Difference:**
```python
# Old:
fed_spn = ProductOverGroups(...)

# New:
products = [ProductOverGroups(...) for combo in combinations]
fed_spn = GlobalSumOfProducts(products, weights)
```

## Testing Plan

**Quick Test (Law School):**
```bash
python tests/benchmarks/test_fedcdh_benchmark_v3.py \
  --datasets law_school \
  --methods fedspn_v \
  --seeds 42 \
  --device cuda
```

**Expected Results:**
- No p-value = 1.000 errors for cross-client edges
- Detect race → UGPA, race → region_first, race → ZFYA
- F1 improvement: 0.222 → 0.60-0.75

**Full Test (Both Datasets):**
```bash
python tests/benchmarks/test_fedcdh_benchmark_v3.py \
  --datasets law_school,sachs \
  --methods fedspn_v \
  --seeds 42 \
  --device cuda
```

## References

**Seng et al. (2025):** "Scaling Probabilistic Circuits via Data Partitioning"
- **Section 3.2:** Product Nodes & Vertical FL (page 4)
- **Assumption 2:** Cluster Independence (page 4)
- **Algorithm 1:** One-Pass Training (page 5, lines 13-19)
- **Definition 2:** Vertical FL (page 4)

**Key Quote (page 4):**
> "In vertical FL, clients hold different feature sets; thus, there is no guarantee
> that the model structure can be shared among clients. [...] A product node assumes
> the random variables of the child distributions to be independent of each other.
> Obviously, this is an unrealistic assumption for vertical FL, where features held
> by different clients might be statistically dependent. Assumption 2 can be exploited
> to capture such dependencies, and a **mixture of products of independent clusters**
> can be formed."
