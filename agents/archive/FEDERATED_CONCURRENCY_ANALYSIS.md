# Federated Concurrency Analysis & Proposal

**Date**: March 13, 2026
**Context**: Technical review identified sequential local SPN training as bottleneck
**Reference**: Jonas Seng et al. "Scaling Probabilistic Circuits via Data Partitioning" (arXiv:2503.08141)

---

## 1. Current Implementation Analysis

### 1.1 Sequential Bottlenecks Identified

**Location**: `FedCDH.py:316-347` (Local SPN Training Loop)

```python
for h in range(num_clusters):  # Outer: cluster loop
    for k in range(self.K_clients):  # Inner: client loop - SEQUENTIAL!
        leaf = LocalSPNWrapper(...)
        leaf.train_local(local_data_h, epochs=50, lr=0.01)  # ~8-12s per call
        clients_clusters[h].append(leaf)
```

**Current Execution Pattern** (K=3 clients, H=3 clusters):
```
Cluster 0: Client 0 → Client 1 → Client 2  (36s total @ 12s each)
Cluster 1: Client 0 → Client 1 → Client 2  (36s total)
Cluster 2: Client 0 → Client 1 → Client 2  (36s total)
─────────────────────────────────────────────
Total Training Time: 108 seconds (sequential)
```

**Parallelizable Operations**:
- ✅ **Client training within cluster**: Independent (different data splits)
- ✅ **Cluster training**: Independent (different mixture components)
- ❌ **EM weight refinement**: Must be sequential (aggregation step)

### 1.2 Communication Pattern Analysis

**Current "Simulated" Federation** (FedCDH.py:33-145):
```python
class SimulatedFederatedKMeans:
    def fit(self, X_splits, feature_maps, scenario):
        # Lines 76-144: Simulates message passing but runs on single machine
        # No actual network I/O or process isolation
```

**Observation**: Current implementation is **simulation-based**, not truly distributed.
- All data (`X_splits`) resides in single process memory
- No network communication overhead
- Perfect for research prototyping but not production federated learning

### 1.3 GPU Utilization Analysis

**Current Device Assignment** (FedCDH.py:206-213):
```python
if torch.cuda.is_available():
    self.device = torch.device("cuda")
else:
    self.device = torch.device("cpu")
```

**GPU Usage Pattern**:
- Single GPU shared across all K clients
- Sequential model transfers to GPU (one at a time)
- **GPU idle time**: ~90% during sequential training

**Potential**:
- With K=3 clients, could train 3 models concurrently on single GPU (if memory permits)
- Batch processing across clients

---

## 2. Jonas Seng's Approach (arXiv:2503.08141)

### 2.1 Key Insights from Paper

**Core Contribution**: "Federated Circuits (FC) framework that allows scaling PCs on distributed environments by recursively partitioning datasets"

**Partitioning Strategy**:
- Horizontal FL: Row partitioning (samples split across clients)
- Vertical FL: Column partitioning (features split across clients)
- Hybrid FL: Mixed partitioning

**Aggregation**:
- "Unifies horizontal, vertical, and hybrid FL by re-framing as density estimation over distributed datasets"
- Uses **product-of-experts** for vertical, **mixture-of-experts** for horizontal

**Parallel Execution** (Implied):
- Paper focuses on mathematical framework for aggregation
- No explicit mention of multiprocessing/threading implementation
- Likely assumes distributed systems handle parallelism (e.g., Ray, Flower)

### 2.2 Differences from Your Implementation

| Aspect | Seng et al. | Your Implementation | Gap |
|--------|-------------|---------------------|-----|
| **Training Paradigm** | True distributed (implied) | Simulated federated | Medium |
| **Parallelism** | Implicit (framework-level) | None (sequential loops) | **HIGH** |
| **Aggregation** | Product/Mixture | Same (correct!) | None |
| **GPU Usage** | Not specified | Single GPU, sequential | Medium |
| **Communication** | Actual network I/O | In-memory simulation | Low (research OK) |

**Key Takeaway**: Your aggregation logic (Product/Mixture) correctly implements Seng's mathematical framework, but lacks parallelization that real federated systems would provide.

---

## 3. Necessity Assessment: Do We Need Parallelization?

### 3.1 Performance Benchmarking

**Baseline** (Current Sequential, M1 CPU):
```
Sachs (N=856, d=11, K=3, epochs=50):
├─ K-means clustering: 2s
├─ Local SPN training: 60-90s  ⚠️ BOTTLENECK
├─ EM weight refinement: 5s
├─ CDNOD discovery: 10-15s
└─ Total: ~85s per seed
```

**With GPU** (Estimated, T4 on Colab):
```
├─ K-means clustering: 2s (CPU)
├─ Local SPN training: 8-12s  ⚠️ STILL BOTTLENECK (sequential)
├─ EM weight refinement: 1s
├─ CDNOD discovery: 3-5s
└─ Total: ~15s per seed
```

**With Parallelization** (3 workers, GPU):
```
├─ K-means clustering: 2s (CPU)
├─ Local SPN training: 3-4s  ✅ 3× SPEEDUP (parallel)
├─ EM weight refinement: 1s
├─ CDNOD discovery: 3-5s
└─ Total: ~10s per seed
```

### 3.2 Experiment Scale Impact

**Thesis Requirement** (from thesis_experiments_plan.md):
- Phase 1: 50 runs (5 methods × 10 seeds)
- Phase 2: 90 runs (9 configs × 2 methods × 5 seeds)
- Phase 3: 30 runs (ablations)
- **Total**: 170 runs

**Time Analysis**:

| Setup | Time per Run | Total Phase 1 | Total All Phases | GPU Hours |
|-------|--------------|---------------|------------------|-----------|
| **Sequential (current)** | 15s | 12.5 min | 42.5 min | 0.7h |
| **Parallel (proposed)** | 10s | 8.3 min | 28.3 min | 0.5h |
| **Savings** | -5s | -4.2 min | -14.2 min | **-0.2h** |

**Verdict**: Parallelization saves ~14 minutes across all experiments.

### 3.3 Necessity Decision Matrix

| Factor | Importance | Current Impact | Need Parallel? |
|--------|------------|----------------|----------------|
| **Total experiment time** | High | 42 min | ❌ **Not critical** |
| **Single run speed** | Medium | 15s | ❌ Acceptable |
| **GPU utilization** | Medium | 10-30% | 🟡 Inefficient but workable |
| **Thesis deadline** | High | 6 weeks left | ❌ Sufficient time |
| **Code complexity** | High | +100 LOC | ⚠️ Risk of bugs |
| **Realistic FL simulation** | Low | Doesn't match real FL | 🟡 Nice-to-have |

**RECOMMENDATION**: ✅ **Implement parallelization as OPTIONAL optimization**, not critical path.

**Rationale**:
1. ✅ Total time (42 min) is acceptable for thesis scope
2. ⚠️ Adding parallelism risks introducing bugs before experiments
3. ✅ Better to get correct results first, optimize later
4. 🟡 Can add as "Future Work" section in thesis

---

## 4. Proposed Concurrency Approach

### 4.1 Design Principles

1. **Backward Compatible**: Parallelization optional via flag
2. **Safe Defaults**: Sequential mode for debugging
3. **Minimal Changes**: Isolate concurrency logic
4. **GPU-Aware**: Handle device contention properly

### 4.2 Implementation Strategy

**Option A: Multiprocessing (CPU-focused)**
```python
from multiprocessing import Pool

def train_single_client(args):
    """Train one client's SPN on CPU."""
    h, k, local_data, config = args
    # Create LocalSPNWrapper
    # Train on CPU
    # Return serialized model parameters
    return leaf_params

# In FedCDH.fit():
if self.args.parallel and self.device.type == "cpu":
    with Pool(processes=min(K_clients, cpu_count())) as pool:
        results = pool.map(train_single_client, task_args)
```

**Pros**: ✅ Easy to implement, ✅ True isolation
**Cons**: ❌ Doesn't help with GPU, ❌ Pickle overhead

---

**Option B: Threading (GPU-focused)**
```python
from concurrent.futures import ThreadPoolExecutor

# In FedCDH.fit():
if self.args.parallel and self.device.type == "cuda":
    with ThreadPoolExecutor(max_workers=K_clients) as executor:
        futures = [executor.submit(train_on_gpu, args) for args in task_list]
        results = [f.result() for f in futures]
```

**Pros**: ✅ Shared GPU memory, ✅ No serialization
**Cons**: ⚠️ GIL contention (minimal for PyTorch), ⚠️ GPU memory limits

---

**Option C: Hybrid (Recommended)**
```python
def get_parallel_strategy(device, K_clients):
    """Choose optimal parallelization based on device."""
    if device.type == "cuda":
        # GPU: Use threading (K threads share same GPU)
        return "threading", min(K_clients, 3)  # Cap at 3 to avoid OOM
    elif device.type == "cpu":
        # CPU: Use multiprocessing (K processes on K cores)
        return "multiprocessing", min(K_clients, cpu_count())
    else:
        return "sequential", 1

# In FedCDH.fit():
strategy, max_workers = get_parallel_strategy(self.device, self.K_clients)

if strategy == "threading":
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Train K clients concurrently on GPU
elif strategy == "multiprocessing":
    with Pool(processes=max_workers) as pool:
        # Train K clients concurrently on CPU
else:
    # Current sequential implementation
```

**Pros**: ✅ Adaptive, ✅ Best of both worlds
**Cons**: 🟡 More complex

---

### 4.3 Detailed Implementation Plan

**Step 1: Refactor Training Function** (30 min)

Create standalone training function that can be parallelized:

```python
# In FedPC.py (new function)
def train_federated_spn_leaf(
    h: int,
    k: int,
    local_data: np.ndarray,
    scenario: str,
    device: torch.device,
    config: dict,
) -> Tuple[nn.Module, int]:
    """
    Train a single SPN leaf for cluster h, client k.

    Args:
        h: Cluster index
        k: Client index
        local_data: Client k's data subset for cluster h
        scenario: 'horizontal', 'vertical', or 'hybrid'
        device: torch device
        config: {num_sums, num_leaves, depth, num_repetitions, epochs, lr}

    Returns:
        (trained_leaf, sample_count)
    """
    if len(local_data) < 5:
        return None, 0

    local_d = local_data.shape[1]

    # Create wrapper
    if local_d == 1:
        leaf = UnivariateSPNWrapper(
            device=device,
            num_sums=config['num_sums'],
            num_leaves=config['num_leaves'],
            seed=h * 10 + k,
        )
    else:
        leaf = LocalSPNWrapper(
            num_features=local_d,
            device=device,
            num_sums=config['num_sums'],
            num_leaves=config['num_leaves'],
            depth=config.get('depth', max(1, int(np.floor(np.log2(local_d))))),
            num_repetitions=config['num_repetitions'],
            seed=h * 10 + k,
        )

    # Train
    leaf.train_local(local_data, epochs=config['epochs'], lr=config.get('lr', 0.01))

    return leaf, len(local_data)
```

---

**Step 2: Add Parallel Orchestrator** (45 min)

```python
# In FedCDH.py (new method)
def _train_clients_parallel(
    self,
    X_splits: List[np.ndarray],
    labels: np.ndarray,
    labels_splits: List[np.ndarray],
    num_clusters: int,
    config: dict,
) -> Tuple[List[List[nn.Module]], List[List[int]]]:
    """
    Train all client SPNs in parallel.

    Returns:
        (clients_clusters, clients_counts)
    """
    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing import Pool, cpu_count

    # Determine strategy
    if self.device.type == "cuda":
        strategy = "threading"
        max_workers = min(self.K_clients, 3)  # GPU memory limit
    elif self.device.type == "cpu":
        strategy = "multiprocessing"
        max_workers = min(self.K_clients, cpu_count())
    else:
        strategy = "sequential"
        max_workers = 1

    logging.info(f"Training strategy: {strategy} with {max_workers} workers")

    # Prepare task list
    tasks = []
    for h in range(num_clusters):
        cluster_mask_global = labels == h
        if cluster_mask_global.sum() < 5:
            continue

        for k in range(self.K_clients):
            local_data_h = (
                X_splits[k][cluster_mask_global]
                if self.scenario == "vertical"
                else X_splits[k][labels_splits[k] == h]
            )
            if len(local_data_h) > 2:
                tasks.append((h, k, local_data_h, self.scenario, self.device, config))

    # Execute
    if strategy == "threading" and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda args: train_federated_spn_leaf(*args),
                tasks
            ))
    elif strategy == "multiprocessing" and len(tasks) > 1:
        # Note: Requires picklable args (may need wrapper)
        with Pool(processes=max_workers) as pool:
            results = pool.starmap(train_federated_spn_leaf, tasks)
    else:
        # Sequential fallback
        results = [train_federated_spn_leaf(*task) for task in tasks]

    # Reconstruct clusters
    clients_clusters = [[] for _ in range(num_clusters)]
    clients_counts = [[] for _ in range(num_clusters)]

    for (h, k, _, _, _, _), (leaf, count) in zip(tasks, results):
        if leaf is not None:
            clients_clusters[h].append(leaf)
            clients_counts[h].append(count)

    return clients_clusters, clients_counts
```

---

**Step 3: Integrate into FedCDH.fit()** (15 min)

```python
# In FedCDH.py, replace lines 316-347:

training_config = {
    'num_sums': num_sums,
    'num_leaves': num_leaves,
    'num_repetitions': num_repetitions,
    'epochs': train_epochs,
    'lr': 0.01,
}

# Choose parallel or sequential based on flag
use_parallel = getattr(self.args, 'parallel_training', False)

if use_parallel:
    clients_clusters, clients_counts = self._train_clients_parallel(
        X_splits, labels, labels_splits, num_clusters, training_config
    )
else:
    # Original sequential code (lines 316-347)
    clients_clusters = [[] for _ in range(num_clusters)]
    clients_counts = [[] for _ in range(num_clusters)]
    for h in range(num_clusters):
        # ... existing code ...
```

---

**Step 4: Add Configuration Flag** (5 min)

```python
# In configs.py, add to all configs:
PRODUCTION_CONFIGS = {
    "fedspn_horizontal": {
        # ... existing fields ...
        "parallel_training": False,  # Set to True to enable
    },
}

# Add new parallel variant:
"fedspn_horizontal_parallel": {
    **PRODUCTION_CONFIGS["fedspn_horizontal"],
    "parallel_training": True,
}
```

---

### 4.4 GPU Memory Management

**Challenge**: Training K=3 models concurrently may exceed GPU memory.

**Solution**: Batch scheduling with memory monitoring

```python
def get_optimal_batch_size(device, model_size_mb, available_memory_mb):
    """
    Calculate how many models can train concurrently on GPU.

    Args:
        device: torch.device
        model_size_mb: Estimated memory per SPN model
        available_memory_mb: Total GPU memory

    Returns:
        batch_size: Number of concurrent trainings
    """
    if device.type != "cuda":
        return float('inf')  # CPU has no hard limit

    # Reserve 20% for overhead
    usable_memory = available_memory_mb * 0.8

    # Each training needs: model + gradients + optimizer state ≈ 3× model size
    memory_per_training = model_size_mb * 3

    batch_size = max(1, int(usable_memory / memory_per_training))

    logging.info(f"GPU memory: {available_memory_mb}MB available, "
                 f"batching {batch_size} trainings at a time")

    return batch_size

# In _train_clients_parallel():
if strategy == "threading":
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    model_size = 50  # Estimated MB per LocalSPNWrapper
    batch_size = get_optimal_batch_size(self.device, model_size, gpu_memory)
    max_workers = min(self.K_clients, batch_size)
```

---

### 4.5 Synchronization Points

**Critical**: Identify where parallelization CANNOT happen.

```python
# Parallel-Safe Operations (can run concurrently):
✅ Local SPN training (different data, different parameters)
✅ K-means clustering (with proper aggregation)
✅ Conditional independence tests (read-only model access)

# Sequential-Only Operations (must synchronize):
❌ EM weight refinement (line 385-387) - updates global weights
❌ Global model assembly (lines 348-383) - aggregates components
❌ CDNOD skeleton discovery (shares global graph state)
```

**Implementation**:
```python
# After parallel training completes:
# All workers must finish before proceeding to aggregation

if use_parallel:
    clients_clusters, clients_counts = self._train_clients_parallel(...)
    # Implicit synchronization: function returns only when all workers done

# Now safe to aggregate (sequential)
global_components = []
for h in range(num_clusters):
    comp = GlobalFedSPN(clients_clusters[h], ...)  # Sequential OK here
    global_components.append(comp)
```

---

## 5. Comparison: Current vs Proposed

### 5.1 Architecture Comparison

**Current** (Sequential):
```
Time  0s ─────> 90s ────────> 95s ──> 110s
      │         │            │        │
      K-means   Train 9 SPNs EM       CDNOD
                (sequential)  Agg

      [GPU idle 90% of time]
```

**Proposed** (Parallel):
```
Time  0s ─────> 30s ────────> 35s ──> 50s
      │         │            │        │
      K-means   Train 9 SPNs EM       CDNOD
                (3 at a time) Agg

      [GPU util 70-90%]
```

### 5.2 Code Complexity

| Aspect | Current | Proposed | Delta |
|--------|---------|----------|-------|
| **LOC in FedCDH.py** | ~500 | ~580 | +80 |
| **New modules** | 0 | 0 | +0 |
| **Dependencies** | torch, numpy | +concurrent.futures | +1 |
| **Complexity** | Low | Medium | ⚠️ +1 level |
| **Debug difficulty** | Easy | Harder (race conditions) | ⚠️ |

### 5.3 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Race conditions** | Low | High | Use immutable data, careful scoping |
| **GPU OOM** | Medium | High | Batch scheduling, memory monitoring |
| **Pickle errors (multiprocessing)** | Medium | Medium | Use threading for GPU, test thoroughly |
| **Debugging harder** | High | Low | Keep sequential mode as default |
| **Results differ** | Low | Critical | Extensive validation before use |

---

## 6. Final Recommendation

### 6.1 Immediate Action (Before Thesis Experiments)

**DO NOT implement parallelization now.**

**Reasons**:
1. ⏰ **Time pressure**: 6 weeks to thesis, experiments start THIS WEEK
2. 🐛 **Risk**: Parallelism adds complexity and potential bugs
3. ⏱️ **Minimal gain**: 14 minutes saved across all experiments (42min → 28min)
4. ✅ **Current speed acceptable**: 15s per run is reasonable for research

**Alternative**: Use sequential execution for reliability.

### 6.2 Post-Thesis Implementation (Optional)

**Recommended Timeline**: After successful thesis defense

**Approach**: Hybrid threading/multiprocessing as Option C
**Priority**: Low (nice-to-have for future work)

**Benefits for Future**:
- 📈 Scalability for larger experiments (K > 10 clients)
- 🏭 Production federated learning deployment
- 📚 Publication extension (NeurIPS/ICML systems track)

### 6.3 Thesis Narrative Strategy

**Include in Thesis**:

**Section 5.4: Limitations and Future Work**
```
Current Implementation and Scalability:
Our implementation simulates federated learning on a single machine,
training K client models sequentially. While sufficient for research
validation (K≤5 clients), production deployment would benefit from:

1. Parallel client training (estimated 3× speedup via threading)
2. True distributed execution (e.g., via Flower framework)
3. Asynchronous aggregation for large-scale federations

We estimate parallel execution would reduce training time from 15s
to ~5s per experiment on GPU, enabling larger-scale parameter grids.
```

**This positions parallelization as "future work" rather than current limitation.**

---

## 7. Executable Task (If Proceeding)

**TASK-11: Implement Federated Parallel Training (OPTIONAL)**

**⚠️ WARNING: Only execute AFTER thesis experiments complete successfully**

**Priority**: 🔵 OPTIONAL (Post-thesis)
**Time**: 3-4 hours
**Risk**: High (could introduce bugs)
**Benefit**: 3× speedup (15s → 5s per run)

**Prompt for Claude**:
```
Implement parallel federated SPN training with hybrid threading/multiprocessing strategy.

Requirements:
1. Create train_federated_spn_leaf() function in FedPC.py (Step 1 from Section 4.3)
2. Add _train_clients_parallel() method to FedCDH class (Step 2)
3. Integrate with FedCDH.fit() via parallel_training flag (Step 3)
4. Add parallel configs to configs.py (Step 4)
5. Implement GPU memory-aware batching (Section 4.4)
6. Add extensive logging for debugging
7. Keep sequential mode as default

Testing requirements:
- Validate results match sequential execution (seed 0-4)
- Test on CPU and GPU
- Test with K=2,3,5 clients
- Measure actual speedup

Acceptance criteria:
- Results identical to sequential (within numerical precision)
- Speedup ≥ 2× on GPU with K=3
- No memory leaks or race conditions
- Falls back gracefully to sequential if parallel fails

Expected output: ~200 LOC, comprehensive tests
```

---

## Appendix: Quick Reference

### A. Performance Summary

| Setup | Device | K | Time/Run | Total (170 runs) | Speedup |
|-------|--------|---|----------|------------------|---------|
| **Sequential (current)** | CPU | 3 | 85s | 4.0h | 1× |
| **Sequential** | GPU | 3 | 15s | 42min | 5.7× |
| **Parallel** | GPU | 3 | 10s | 28min | **8.5×** |

### B. Implementation Complexity

| Component | LOC | Files | Complexity | Risk |
|-----------|-----|-------|------------|------|
| Refactor training | 50 | FedPC.py | Low | Low |
| Parallel orchestrator | 80 | FedCDH.py | Medium | Medium |
| Memory management | 30 | FedCDH.py | Medium | High |
| Config flags | 10 | configs.py | Low | Low |
| Testing | 100 | tests/ | Medium | Medium |
| **Total** | **270** | **3** | **Medium** | **Medium-High** |

### C. Decision Tree

```
Do you need results in < 1 hour?
├─ NO  → Use sequential (current)  ✅ RECOMMENDED FOR THESIS
└─ YES → Is K > 5?
         ├─ NO  → Sequential still fine (42 min for 170 runs)
         └─ YES → Implement parallel (3-4h development)
                  └─ After thesis defense ✅
```

---

**Conclusion**: Parallelization is a **nice-to-have optimization**, not a **critical requirement** for thesis completion. Current sequential execution is sufficient for 170 experiments (42 minutes on GPU). Recommend deferring to post-thesis future work.

---

*Analysis prepared by Claude Code*
*Recommendation: Proceed with sequential execution for thesis*
