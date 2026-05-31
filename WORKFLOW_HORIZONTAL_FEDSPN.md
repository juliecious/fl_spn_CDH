# Workflow: Horizontal FedSPN-CDH with Structure Voting

**Complete Pipeline for Asia Benchmark (Horizontal Mode)**

---

## Phase 1: Data Preparation & Partitioning

```
Input: X_train (999 samples, 8 features)

┌─────────────────────────────────────────┐
│ Horizontal Data Partitioning            │
│ - Augment with client IDs (column 8)   │
│ - Split by samples (not features)      │
└─────────────────────────────────────────┘
                    ↓
X_aug = [X_train | client_id]  # (999, 9)

Client 0: X_aug[0:333, :]     = (333, 9)  # client_id = 0
Client 1: X_aug[333:666, :]   = (333, 9)  # client_id = 1
Client 2: X_aug[666:999, :]   = (333, 9)  # client_id = 2
```

**Key Point**: Each client sees ALL 8 features + client ID (column 8), but only their subset of samples.

---

## Phase 2: Local SPN Training (Per-Client)

```
┌─────────────────────────────────────────┐
│ FOR EACH CLIENT k = 0, 1, 2:           │
│                                         │
│ 1. Local K-means clustering             │
│    - K_local = 2 (per client)          │
│    - Split data into 2 clusters        │
│                                         │
│ 2. Train SPN per cluster                │
│    - Cluster 0: ~315 samples → SPN_k0  │
│    - Cluster 1: ~18 samples → SPN_k1   │
│                                         │
│ 3. Build local mixture                  │
│    LocalSPN_k = w_0 * SPN_k0 +         │
│                 w_1 * SPN_k1           │
│                                         │
│ Output: 3 local SPNs (one per client)  │
└─────────────────────────────────────────┘

LocalSPN_0: P(X_0, ..., X_7, U | Client 0 data)
LocalSPN_1: P(X_0, ..., X_7, U | Client 1 data)
LocalSPN_2: P(X_0, ..., X_7, U | Client 2 data)

where U = augmented variable (client ID)
```

**Key Point**: Local SPNs are trained on **9D data** (8 features + client ID). Each SPN learns the distribution for its client's data.

---

## Phase 3: Structure Voting (Uses LOCAL SPNs)

```
┌─────────────────────────────────────────────────────────┐
│ Structure Voting: Extract Dependency Graphs             │
│                                                          │
│ FOR EACH CLIENT k = 0, 1, 2:                           │
│                                                          │
│   Use LocalSPN_k to perform CI tests:                  │
│   - Test all pairs (i, j) where i,j ∈ {0,1,...,7}    │
│   - Conditioning set: Z = [8] (augmented variable!)   │
│   - CI Test: X_i ⊥ X_j | U  using LocalSPN_k         │
│                                                          │
│   Example:                                              │
│     Test: X_0 ⊥ X_1 | U                                │
│     Method: Likelihood ratio using LocalSPN_k          │
│     Result: p-value > 0.05 → INDEPENDENT → no edge     │
│             p-value < 0.05 → DEPENDENT → add edge      │
│                                                          │
│   Output: Local dependency graph G_k (undirected)       │
│           G_0: 18 edges                                 │
│           G_1: 21 edges                                 │
│           G_2: 19 edges                                 │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│ Consensus Aggregation (Voting)                          │
│                                                          │
│   Edge voting: Count how many clients found each edge   │
│   Threshold: 2/3 clients (66.7% confidence)            │
│                                                          │
│   Example:                                              │
│     Edge (0,1): Found by clients 0,1,2 → 100% → Include│
│     Edge (2,5): Found by clients 0,1 → 66.7% → Include │
│     Edge (4,7): Found by client 0 only → 33.3% → Exclude│
│                                                          │
│   Output: Consensus graph G_consensus                   │
│           22 edges (93.9% avg confidence)              │
└─────────────────────────────────────────────────────────┘

initial_skeleton = adjacency_matrix(G_consensus)  # (8, 8)
```

**Key Point**: Structure voting uses **LOCAL SPNs**, conditioning on U (augmented variable). This happens BEFORE building global SPN.

**Your Understanding**: ❌ "then structure voting" after global mixture
**Correct Order**: Structure voting happens BEFORE global mixture

---

## Phase 4: Global SPN Training (Mixture of Local SPNs)

```
┌─────────────────────────────────────────────────────────┐
│ Build Global Mixture Model                              │
│                                                          │
│   GlobalSPN = (1/3) * LocalSPN_0 +                     │
│               (1/3) * LocalSPN_1 +                     │
│               (1/3) * LocalSPN_2                       │
│                                                          │
│   This is a MIXTURE, not a separate training!          │
│   Weights = [1/3, 1/3, 1/3] (equal for horizontal)    │
│                                                          │
│   GlobalSPN(x) = Σ_k w_k * LocalSPN_k(x)              │
│                                                          │
│   Output: GlobalSPN for P(X_0, ..., X_7, U)           │
└─────────────────────────────────────────────────────────┘

GlobalSPN: Used for CI testing in CDNOD
```

**Key Point**: Global SPN is a **weighted mixture** of local SPNs, not trained from scratch. Weights are equal (1/3 each) for horizontal mode.

---

## Phase 5: (Optional) Covariance Tensor Derivation

```
┌─────────────────────────────────────────────────────────┐
│ Derive Covariance Tensor from GlobalSPN (Optional)      │
│                                                          │
│   For orientation via FICP, we can optionally compute:  │
│                                                          │
│   Cov[X_i, X_j | U=k] for all i,j,k                    │
│                                                          │
│   Method:                                               │
│   - Sample from GlobalSPN conditioned on U=k           │
│   - Compute empirical covariance matrices              │
│   - Store in tensor: (K_clients, d, d)                │
│                                                          │
│   This is used for faster FICP orientation             │
└─────────────────────────────────────────────────────────┘

covariance_tensor: (3, 8, 8)  # Optional optimization
```

**Key Point**: This is **optional** and used for orientation optimization. Not required for CI testing.

---

## Phase 6: CDNOD - Skeleton Discovery with FCIT

```
┌─────────────────────────────────────────────────────────┐
│ CDNOD Algorithm (PC with SPN-based CI testing)          │
│                                                          │
│ Input:                                                  │
│   - X_train: (999, 8) - original features             │
│   - initial_skeleton: (8, 8) - from structure voting  │
│   - GlobalSPN: for CI testing                          │
│                                                          │
│ Initialization:                                         │
│   - Create CausalGraph with 8 variables               │
│   - CLEAR graph: graph[:] = 0                          │
│   - Load initial_skeleton: 22 edges                    │
│                                                          │
│ Stage 1: Skeleton Discovery                            │
│   PC Algorithm with SPN-based FCIT:                    │
│                                                          │
│   FOR depth d = 0, 1, 2, 3:                           │
│     FOR each edge (i, j) in current skeleton:         │
│       FOR each conditioning set Z of size d:          │
│                                                          │
│         CI Test: X_i ⊥ X_j | Z                        │
│                                                          │
│         Method: Functional CI Test (FCIT)             │
│         ┌───────────────────────────────────┐        │
│         │ 1. Compute log-likelihoods:        │        │
│         │    ll_xyz = log P(X_i, X_j, Z)    │        │
│         │    ll_xz  = log P(X_i, Z)         │        │
│         │    ll_yz  = log P(X_j, Z)         │        │
│         │    ll_z   = log P(Z)              │        │
│         │                                    │        │
│         │ 2. Compute CMI (using GlobalSPN): │        │
│         │    CMI = ll_xyz - ll_xz - ll_yz + ll_z │  │
│         │                                    │        │
│         │ 3. Permutation test:               │        │
│         │    - Shuffle X_j                   │        │
│         │    - Compute CMI_null              │        │
│         │    - p-value = P(CMI_null > CMI_obs) │    │
│         └───────────────────────────────────┘        │
│                                                          │
│         If p-value > α (0.05): Remove edge (i,j)      │
│                                                          │
│   Output: Refined skeleton (e.g., 8-12 edges)          │
│                                                          │
│ Stage 2: Surrogate Node (Skipped if exclude_augmented) │
│   (Would add edges to/from augmented variable)         │
│                                                          │
└─────────────────────────────────────────────────────────┘

Refined Skeleton: ~8-12 edges (depends on data)
```

**Key Point**: FCIT uses **GlobalSPN** to compute likelihoods for CI testing. This is done WITHOUT conditioning on U (augmented variable is excluded from graph).

**Your Understanding**: ✅ "use SPN to perform FCIT" - Correct!

---

## Phase 7: Edge Orientation with FICP

```
┌─────────────────────────────────────────────────────────┐
│ Mechanism Invariance Orientation (FICP)                 │
│                                                          │
│ Input: Undirected skeleton from Stage 1                │
│                                                          │
│ For each undirected edge i -- j:                        │
│                                                          │
│   Goal: Determine direction (i → j or j → i)          │
│                                                          │
│   Principle: Federated Independent Change Principle     │
│   "If i → j, then P(X_j | X_i) is invariant across    │
│    domains U. If not invariant, try j → i."           │
│                                                          │
│   Method: Conditional Mutual Information (CMI)         │
│   ┌───────────────────────────────────────────┐      │
│   │ For direction i → j:                       │      │
│   │   Score_i→j = I(X_j; U | X_i)             │      │
│   │             = CMI using GlobalSPN          │      │
│   │                                            │      │
│   │ For direction j → i:                       │      │
│   │   Score_j→i = I(X_i; U | X_j)             │      │
│   │                                            │      │
│   │ Decision:                                  │      │
│   │   If Score_i→j < Score_j→i:               │      │
│   │     Direction is i → j (more invariant)   │      │
│   │   Else:                                    │      │
│   │     Direction is j → i                    │      │
│   └───────────────────────────────────────────┘      │
│                                                          │
│   Implementation uses GlobalSPN + X_aug_splits:        │
│   - Split data by client                               │
│   - Compute CMI for each direction                     │
│   - Choose direction with lower CMI (more invariant)   │
│                                                          │
│   Output: Directed edges (DAG/CPDAG)                   │
└─────────────────────────────────────────────────────────┘

Final Graph: Directed Acyclic Graph (DAG)
```

**Key Point**: FICP uses **GlobalSPN** to compute CMI scores for orientation. This explicitly uses U (augmented variable) to measure invariance.

**Your Understanding**: ✅ "use SPN to perform... FICP" - Correct!

---

## Summary: Correcting Your Understanding

### Your Understanding:
> "we first split the data, train horizontal SPN in 3 diff. clients, build global mixture **then structure voting**, then you derive covariance from trained global SPN, use SPN to perform FCIT and FICP"

### Corrections:

#### ❌ Incorrect Order
**Your order**: Local SPNs → Global Mixture → Structure Voting
**Correct order**: Local SPNs → **Structure Voting** → Global Mixture

**Why**: Structure voting uses **local SPNs** to extract dependency graphs, then aggregates them. The global mixture is built AFTER structure voting.

#### ⚠️ Minor Clarification
**Covariance derivation**: Optional optimization for FICP, not always done

#### ✅ Correct Parts
- Split data (horizontal partitioning)
- Train local SPNs on 3 clients
- Use GlobalSPN for FCIT (CI testing)
- Use GlobalSPN for FICP (orientation)

---

## Complete Correct Workflow

```
1. Data Split
   ↓
2. Train Local SPNs (3 clients, on 9D data)
   ↓
3. Structure Voting (using Local SPNs, conditioning on U)
   ↓ (produces initial_skeleton: 22 edges)
   ↓
4. Build Global SPN (mixture of local SPNs)
   ↓
5. (Optional) Derive Covariance Tensor
   ↓
6. CDNOD Skeleton Discovery (using Global SPN for FCIT)
   ↓ (refines skeleton to ~8-12 edges)
   ↓
7. Edge Orientation (using Global SPN for FICP)
   ↓
8. Final DAG
```

---

## Key Distinctions

### Local SPNs vs Global SPN

| Aspect | Local SPNs | Global SPN |
|--------|-----------|------------|
| When | Phase 2 | Phase 4 |
| Training | Per-client on client data | Mixture of local SPNs |
| Data | Client-specific (333 samples) | All data (999 samples) |
| Used for | Structure voting | CI testing + Orientation |
| Count | 3 (one per client) | 1 (mixture model) |

### Structure Voting vs CDNOD

| Aspect | Structure Voting | CDNOD |
|--------|-----------------|--------|
| When | Phase 3 (before global SPN) | Phase 6 (after global SPN) |
| Uses | **Local SPNs** | **Global SPN** |
| Conditioning | Z = [8] (always include U) | Z = any subset of {0,...,7} (excludes U) |
| Output | Initial skeleton (22 edges) | Refined skeleton (8-12 edges) |
| Purpose | Bootstrap with high-quality edges | Refine skeleton via PC algorithm |

---

## Why This Design?

1. **Local SPNs capture client-specific patterns** → Good for initial dependency detection
2. **Structure voting aggregates across clients** → Robust consensus skeleton
3. **Global SPN represents full distribution** → Accurate CI testing
4. **FCIT with GlobalSPN** → Correct conditional independence tests
5. **FICP with GlobalSPN** → Invariance-based orientation using domain information

This design combines the strengths of federated learning (privacy, local patterns) with centralized refinement (accurate global distribution).
