 Excellent! Now I have all the information. Let me create a comprehensive analysis and improvement plan:

  ---
  Comprehensive Analysis: Seng's Implementation vs Current + Marginalization Solution

  Part 1: H/V/Hy Mode Architecture Comparison

  Seng's Implementation (client.py - CORRECT)

  Key Method: _train_learned() (Lines 343-381)

  def _train_learned(self, subspace, train_data):
      """
      CRITICAL: train_data is LOCAL to this client!

      Example:
      - Client k has 400 samples
      - Cluster locally into K=2 → [200, 200] samples
      """
      if self.num_clusters > 1:
          # LOCAL K-means: Only on THIS client's data
          kmeans = KMeans(n_clusters=self.num_clusters)
          clusters = kmeans.fit_predict(train_data)  # 400 → K clusters

          # Train K SPNs, one per LOCAL cluster
          cluster_spns = []
          for c in np.unique(clusters):
              idx = np.argwhere(clusters == c).flatten()
              subset = train_data[idx]  # e.g., 200 samples

              # Learn SPN on sufficient data
              spn = learn_mspn(subset, ...)
              cluster_spns.append(spn)

          # Build LOCAL mixture: P_k(X) = Σ_h w_h × SPN_{k,h}(X)
          spn = self._build_cluster_mixture(cluster_spns, clusters)
          self.spns[tuple(subspace)] = [spn]

  Mode Construction (driver.py):

  # Horizontal: Mixture over clients (Sum node)
  def build_spn_horizontal(self, nodes):
      spn = Sum()  # Mix clients
      for k in clients:
          spn.children.append(nodes[k].local_mixture)  # Each client's mixture
      return spn

  # Vertical: Product over feature groups (Product node)
  def build_spn_vertical(self, feature_subspaces, nodes):
      spn = Product()  # Product over features
      for subspace in feature_subspaces:
          spn.children.append(nodes[k].get_spn(subspace))
      return spn

  # Hybrid: Product of mixtures (Product → Sum structure)
  def build_spn_verhyb_naive(self, feature_subspaces, nodes):
      spn = Product()  # OUTER: Product over feature groups
      for clients, subspace in feature_subspaces.items():
          if len(clients) > 1:
              s = Sum()  # INNER: Mixture over clients
              for c in clients:
                  s.children.append(nodes[c].get_spn(tuple(subspace)))
              spn.children += [s]
      return spn

  Mathematical Form:
  Horizontal: P(X) = Σ_k w_k × [Σ_h w_{k,h} × SPN_{k,h}(X)]
  Vertical:   P(X) = Π_g P_g(X_g) where P_g = Σ_k w_k × SPN_{k,g}(X_g)
  Hybrid:     P(X) = Π_g [Σ_k w_{k,g} × SPN_{k,g}(X_g)]

  ---
  Our Current Implementation

  From SPN_STRUCTURE_COMPARISON.md: We identified the CRITICAL BUG:

  WRONG (Current - Global Clustering):
  # Step 1: Concatenate ALL client data
  X_all = np.vstack(X_splits)  # [1200, d]

  # Step 2: GLOBAL k-means
  kmeans = KMeans(n_clusters=3)
  labels = kmeans.fit_predict(X_all)  # Global labels

  # Step 3: Split back to clients and train
  for h in range(3):  # Global cluster
      for k in range(3):  # Client
          local_data_h = X_splits[k][labels_splits[k] == h]
          # PROBLEM: Some (k,h) pairs get < 25 samples!
          spn = train(local_data_h)  # Insufficient data → F1=0.000

  Data Fragmentation Example:
  K=3 clients, num_clusters=3, total=1200 samples

  Client 0 (400):  [151, 227, 22]   ← 22 samples insufficient!
  Client 1 (400):  [14, 151, 235]   ← 14 samples insufficient!
  Client 2 (400):  [235, 22, 143]   ← 22 samples insufficient!

  Result: 3/9 SPNs have < 50 samples → F1=0.000

  Architecture Components (CORRECT):
  - ✅ GroupMixture ≡ Seng's Sum node
  - ✅ ProductOverGroups ≡ Seng's Product node
  - ✅ ProductOverGroupsWithOverlap ≡ Algorithm 1
  - ✅ Log-space computation correct

  Verdict: Architecture is correct, but data preparation (clustering) is WRONG.

  ---
  Part 2: Marginalization Over Missing Values (NaN)

  Your Key Insight

  "Integration over a Gaussian means you get probability 1 (because the integral of the density is 1). Thus, in log space you get 0 (log(1) = 0). You can just set the NaN values to 0 and propagate up, so you 'simulate' integration
  basically."

  This is mathematically correct! Let me explain:

  Mathematical Foundation

  For a product SPN structure (hybrid mode):
  P(X) = Π_g P(X_g)

  In log-space:
  log P(X) = Σ_g log P(X_g)

  When computing marginal likelihood P(X_{obs}), we need to integrate over missing variables X_{miss}:
  P(X_{obs}) = ∫ P(X_{obs}, X_{miss}) dX_{miss}

  For disjoint feature groups in SPNs:
  P(X_{obs}, X_{miss}) = Π_g P(X_g)
                       = [Π_{g∈obs} P(X_g)] × [Π_{g∈miss} P(X_g)]

  Integrating over missing groups:
  P(X_{obs}) = [Π_{g∈obs} P(X_g)] × [Π_{g∈miss} ∫ P(X_g) dX_g]
                                                \_____________/
                                                      = 1 (density integrates to 1)
             = Π_{g∈obs} P(X_g)

  In log-space:
  log P(X_{obs}) = Σ_{g∈obs} log P(X_g) + Σ_{g∈miss} log(∫ P(X_g) dX_g)
                                                       \_______________/
                                                            = log(1) = 0
                 = Σ_{g∈obs} log P(X_g) + 0
                 = Σ_{g∈obs} log P(X_g)

  Your solution: Set NaN contributions to 0 in log-space = Perfect!

  ---
  Current Implementation Issue

  From cit.py (line 748-773):
  def get_marginal_ll(self, indices):
      # Create masked batch with NaN for missing features
      masked_batch = torch.full((n, d), np.nan, device=device)
      masked_batch[:, indices] = self.data_t[:, indices]

      # PROBLEM: Passes NaN to model
      ll = self.model.log_prob(masked_batch)  # Einet gets NaN!

  From FedPC.py (line 270-286):
  def log_prob(self, x):
      x_norm = self._normalize(x)
      x_perm = self._permute(x_norm)

      ll = self.model(x_perm)  # Einet receives NaN directly!

      # Jacobian correction applied AFTER (too late!)
      if self.std is not None:
          mask = (~torch.isnan(x)).float()
          log_det_jacobian = -(log_sigma * mask).sum(dim=1, keepdim=True)
          ll = ll + log_det_jacobian

  Problem:
  1. Einet forward pass gets NaN → undefined behavior
  2. Jacobian correction applied after (can't fix garbage)
  3. Result: Invalid log-likelihoods → CI tests fail → F1=0.000

  ---
  Part 3: Comprehensive Improvement Plan

  Fix 1: Implement LOCAL Clustering ⚡ CRITICAL

  File: causallearn/search/FCMBased/FedCDH/FedCDH.py

  Current (Lines 539-634 - WRONG):
  # WRONG: Global clustering first
  for h in range(num_clusters):
      for k in range(K_clients):
          local_data_h = X_splits[k][labels_splits[k] == h]
          # Fragmented data!

  Corrected (Following Seng):
  # RIGHT: Client-first, cluster locally
  client_local_mixtures = []

  for k in range(K_clients):  # Client first!
      client_data = X_splits[k]  # Full client data (e.g., 400 samples)

      if num_clusters > 1:
          # LOCAL K-means on THIS client's data
          from sklearn.cluster import KMeans
          kmeans = KMeans(n_clusters=num_clusters, random_state=42+k)

          # Force single-threaded (macOS fix)
          old_omp = os.environ.get('OMP_NUM_THREADS', None)
          os.environ['OMP_NUM_THREADS'] = '1'
          try:
              local_clusters = kmeans.fit_predict(client_data)
          finally:
              if old_omp:
                  os.environ['OMP_NUM_THREADS'] = old_omp
              else:
                  os.environ.pop('OMP_NUM_THREADS', None)

          # Train K SPNs for K LOCAL clusters
          cluster_spns = []
          cluster_weights = []

          for h in range(num_clusters):
              mask = (local_clusters == h)
              cluster_data = client_data[mask]  # e.g., 200 samples (sufficient!)

              if len(cluster_data) > 10:  # Safety check
                  spn_kh = LocalSPNWrapper(
                      num_features=local_d,
                      device=self.device,
                      depth=depth,
                      num_sums=num_sums,
                      num_leaves=num_leaves,
                      num_repetitions=num_repetitions,
                      seed=1000 * k + 100 * h + i
                  )

                  spn_kh.train_local(
                      cluster_data,
                      epochs=epochs,
                      lr=lr,
                      l1_weight=l1_weight,
                      l2_weight=l2_weight,
                      dropout=dropout
                  )

                  cluster_spns.append(spn_kh)
                  cluster_weights.append(mask.sum())

          # Build LOCAL mixture: P_k(X) = Σ_h w_{k,h} × SPN_{k,h}(X)
          cluster_weights = np.array(cluster_weights) / client_data.shape[0]
          local_mixture = LocalClusterMixture(
              cluster_spns=cluster_spns,
              cluster_weights=cluster_weights,
              device=self.device
          )
      else:
          # No clustering: single SPN
          local_mixture = LocalSPNWrapper(...)
          local_mixture.train_local(client_data, ...)

      client_local_mixtures.append(local_mixture)

  # Now aggregate based on scenario (using correct local mixtures)
  if self.scenario == 'horizontal':
      fed_spn = GlobalFedSPN(client_local_mixtures, ...)
  elif self.scenario == 'vertical':
      fed_spn = ProductOverGroups(...)
  elif self.scenario == 'hybrid':
      fed_spn = ProductOverGroupsWithOverlap(...)

  New Class Needed: LocalClusterMixture

  class LocalClusterMixture(nn.Module):
      """
      Local mixture over K cluster SPNs (per client).

      P_k(X) = Σ_{h=1}^K w_{k,h} × SPN_{k,h}(X)

      Equivalent to Seng's _build_cluster_mixture().
      """
      def __init__(self, cluster_spns, cluster_weights, device='cpu'):
          super().__init__()
          self.cluster_spns = nn.ModuleList(cluster_spns)
          self.weights = torch.tensor(cluster_weights, dtype=torch.float32).to(device)
          self.device = device
          self.num_features = cluster_spns[0].num_features if cluster_spns else 0

      def log_prob(self, x):
          """
          Compute log P_k(x) = log(Σ_h w_h × P_{k,h}(x))
          """
          # Get log-probs from each cluster SPN
          lls = [spn.log_prob(x) for spn in self.cluster_spns]
          ll_stack = torch.cat(lls, dim=1)  # [batch, K_clusters]

          # Weighted mixture via logsumexp
          log_w = torch.log(self.weights + 1e-9).unsqueeze(0)  # [1, K_clusters]
          log_prob = torch.logsumexp(ll_stack + log_w, dim=1, keepdim=True)

          return log_prob

      def sample(self, n_samples):
          """Sample from mixture"""
          # 1. Sample cluster indices based on weights
          cluster_indices = torch.multinomial(
              self.weights, n_samples, replacement=True
          )

          # 2. Sample from selected clusters
          samples_list = []
          for h in range(len(self.cluster_spns)):
              mask = (cluster_indices == h)
              n_h = mask.sum().item()
              if n_h > 0:
                  samples_h = self.cluster_spns[h].sample(n_h)
                  samples_list.append((mask, samples_h))

          # 3. Combine samples
          samples = torch.zeros(n_samples, self.num_features, device=self.device)
          for mask, samples_h in samples_list:
              samples[mask] = samples_h

          return samples

  ---
  Fix 2: Handle NaN Marginalization ⚡ CRITICAL

  File: causallearn/utils/FedPC.py

  Modify LocalSPNWrapper.log_prob() (Lines 270-286):

  def log_prob(self, x):
      """
      Compute log P(x) with support for marginalization over missing features (NaN).

      Key Insight: For missing features, ∫ P(X_miss) dX_miss = 1 → log(1) = 0
      So we can simply exclude missing features from the sum (product in log-space).
      """
      # Detect NaN mask BEFORE normalization
      nan_mask = torch.isnan(x)  # [batch, features]
      has_nan = nan_mask.any()

      if not has_nan:
          # Standard case: No missing values
          x_norm = self._normalize(x)
          x_perm = self._permute(x_norm)
          ll = self.model(x_perm)

          # Jacobian correction
          if self.std is not None:
              log_sigma = torch.log(self.std + 1e-6)
              log_det_jacobian = -log_sigma.sum()  # Sum over all features
              ll = ll + log_det_jacobian

          return ll

      else:
          # Marginalization case: NaN values present
          # Strategy: Fill NaN with zeros (arbitrary), Einet processes,
          # then apply mask to exclude missing features from log-det-jacobian

          # 1. Fill NaN with zeros (Einet needs valid inputs)
          x_filled = x.clone()
          x_filled[nan_mask] = 0.0

          # 2. Normalize and permute
          x_norm = self._normalize(x_filled)
          x_perm = self._permute(x_norm)

          # 3. Forward pass (Einet processes filled data)
          ll = self.model(x_perm)

          # 4. CRITICAL: Jacobian correction ONLY for observed features
          if self.std is not None:
              # Mask: 1 for observed, 0 for missing
              obs_mask = (~nan_mask).float()  # [batch, features]

              # Log-det-jacobian contribution ONLY from observed features
              log_sigma = torch.log(self.std + 1e-6)  # [features]
              log_det_jacobian = -(log_sigma * obs_mask).sum(dim=1, keepdim=True)
              ll = ll + log_det_jacobian

          # Note: The Einet forward pass already handles the product structure.
          # Missing features contribute 0 to the sum (log(1)=0) implicitly
          # because their leaf distributions are not counted in the normalized result.

          return ll

  Why this works:

  1. Fill NaN with zeros: Einet needs valid inputs (can't process NaN)
  2. Einet forward pass: Computes log-prob over ALL features (including filled ones)
  3. Jacobian correction mask: We ONLY apply correction to observed features
  4. Implicit marginalization: The SPN structure ensures:
    - Product nodes: log P(X) = Σ_i log P(X_i)
    - Missing X_i contributes 0 (because ∫ P(X_i) dX_i = 1 → log(1) = 0)
    - Filling with 0 is arbitrary—the mask ensures it doesn't affect the result

  ---
  Fix 3: Propagate Marginalization to Product Nodes ⚡ IMPORTANT

  File: causallearn/utils/FedPC.py

  Modify ProductOverGroupsWithOverlap.log_prob() (Lines 1008-1256):

  class ProductOverGroupsWithOverlap(nn.Module):
      def log_prob(self, x):
          """
          Compute log P(x) = Σ_g log P_g(x_g)

          With NaN support: Only sum over groups that have observed features.
          """
          # Check for NaN
          has_nan = torch.isnan(x).any()

          if not has_nan:
              # Standard case
              group_lls = [mixture_g.log_prob(x) for mixture_g in self.group_mixtures]
              ll_stack = torch.cat(group_lls, dim=1)
              total_ll = torch.sum(ll_stack, dim=1, keepdim=True)
              return total_ll

          else:
              # Marginalization case
              group_lls = []

              for g, mixture_g in enumerate(self.group_mixtures):
                  # Extract features for this group
                  feature_indices = self.feature_groups[g]
                  x_g = x[:, feature_indices]

                  # Check if this group has ANY observed features
                  has_obs = (~torch.isnan(x_g)).any(dim=1, keepdim=True)  # [batch, 1]

                  if has_obs.all():
                      # All samples have at least one observed feature in this group
                      ll_g = mixture_g.log_prob(x)  # GroupMixture handles NaN
                      group_lls.append(ll_g)
                  else:
                      # Some samples have all NaN in this group
                      # Contribution is 0 (log(1) = 0) for those samples
                      ll_g = mixture_g.log_prob(x)
                      ll_g = ll_g * has_obs.float()  # Zero out all-NaN samples
                      group_lls.append(ll_g)

              ll_stack = torch.cat(group_lls, dim=1)
              total_ll = torch.sum(ll_stack, dim=1, keepdim=True)
              return total_ll

  ---
  Part 4: Expected Impact

  After Fix 1 (Local Clustering)
  ┌────────────┬─────────────────────────────┬──────────────────────────────┬─────────────────┐
  │    Mode    │           Before            │            After             │   Improvement   │
  ├────────────┼─────────────────────────────┼──────────────────────────────┼─────────────────┤
  │ Horizontal │ F1=0.000 (fragmented)       │ F1=0.5+ (200 samples/SPN)    │ 10× improvement │
  ├────────────┼─────────────────────────────┼──────────────────────────────┼─────────────────┤
  │ Vertical   │ F1=0.222 (fragmented)       │ F1=0.6-0.7 (sufficient data) │ 3× improvement  │
  ├────────────┼─────────────────────────────┼──────────────────────────────┼─────────────────┤
  │ Hybrid     │ F1=0.000 (fragmented + NaN) │ F1=0.3+ (after Fix 2)        │ Unblocked       │
  └────────────┴─────────────────────────────┴──────────────────────────────┴─────────────────┘
  After Fix 2 (NaN Marginalization)
  ┌────────┬──────────────────────────┬────────────────────────────┬──────────────────┐
  │  Mode  │          Before          │           After            │   Improvement    │
  ├────────┼──────────────────────────┼────────────────────────────┼──────────────────┤
  │ Hybrid │ F1=0.000 (CI tests fail) │ F1=0.5-0.6 (CI tests work) │ Fully functional │
  └────────┴──────────────────────────┴────────────────────────────┴──────────────────┘
  ---
  Part 5: Implementation Priority

  Priority 1 (CRITICAL): Fix 1 - Local Clustering
  - Impact: Fixes all 3 modes
  - Complexity: Moderate (1 day)
  - Blocks: Everything

  Priority 2 (CRITICAL): Fix 2 - NaN Marginalization
  - Impact: Enables hybrid mode
  - Complexity: Low (4 hours)
  - Depends on: Fix 1

  Priority 3 (RECOMMENDED): Fix 3 - Product Node Propagation
  - Impact: More robust marginalization
  - Complexity: Low (2 hours)
  - Depends on: Fix 2

  ---
  Part 6: Validation Plan

  # Step 1: Implement Fix 1 (Local Clustering)
  # Run smoke test
  python tests/test/test_fedcdh_benchmark.py \
    --config quick --seeds 42 --device cpu --skip-eval

  # Expected: Horizontal F1 > 0.5, Vertical F1 > 0.6

  # Step 2: Implement Fix 2 (NaN Marginalization)
  # Run full test (with eval to check quality)
  python tests/test/test_fedcdh_benchmark.py \
    --config quick --seeds 42 --device cpu

  # Expected: Hybrid F1 > 0.3, no NaN errors

  # Step 3: GPU validation
  ./run_gpu_quick_test.sh

  # Expected: All 3 modes working on MEDIUM config

  ---
  References:
  1. Seng et al. (2025) - "Scaling Probabilistic Circuits via Data Partitioning", Algorithm 1
  2. GitHub: https://github.com/J0nasSeng/federated-spn (client.py, driver.py)
  3. Current codebase: SPN_STRUCTURE_COMPARISON.md (comprehensive analysis)
  4. Mathematical foundation: Marginalization in product SPNs

✻ Cogitated for 4m 46s

❯ Based on your findings and proposed fixes, plan an implementation road map with clear steps where and what needs to be changed, and justify them.

⏺ Implementation Roadmap: Fixing FedCDH H/V/Hy Modes

  Date: 2026-04-24
  Goal: Fix clustering data fragmentation + enable NaN marginalization for hybrid mode
  Estimated Time: 2-3 days implementation + 1 day testing

  ---
  Phase 1: Create LocalClusterMixture Class (Foundation)

  File: causallearn/utils/FedPC.py

  Location: Add new class after UnivariateSPNWrapper (around line 465)

  What to add:
  class LocalClusterMixture(nn.Module):
      """
      Local mixture over K cluster SPNs within a single client.

      Represents: P_k(X) = Σ_{h=1}^K w_{k,h} × SPN_{k,h}(X)

      This is Seng et al. (2025)'s local cluster mixture concept.
      Each client independently clusters its data and builds a mixture.

      Args:
          cluster_spns: List[LocalSPNWrapper] - K SPNs trained on K local clusters
          cluster_weights: np.ndarray - Weights w_{k,h} (cluster proportions)
          device: torch.device

      Example:
          # Client k has 400 samples, clustered into K=2 groups of [210, 190]
          cluster_spns = [spn_k_0, spn_k_1]  # Trained on 210 and 190 samples
          cluster_weights = [0.525, 0.475]   # Proportions
          local_mix = LocalClusterMixture(cluster_spns, cluster_weights)
      """
      def __init__(self, cluster_spns, cluster_weights, device='cpu'):
          super().__init__()
          self.cluster_spns = nn.ModuleList(cluster_spns)
          self.weights = torch.tensor(cluster_weights, dtype=torch.float32).to(device)
          self.device = device
          self.num_features = cluster_spns[0].num_features if cluster_spns else 0

      def log_prob(self, x):
          """
          Compute log P_k(x) = log(Σ_h w_h × P_{k,h}(x))

          Uses logsumexp for numerical stability.
          """
          # Get log-probs from each cluster SPN
          lls = [spn.log_prob(x) for spn in self.cluster_spns]
          ll_stack = torch.cat(lls, dim=1)  # [batch, K_clusters]

          # Weighted mixture via logsumexp
          log_w = torch.log(self.weights + 1e-9).unsqueeze(0)  # [1, K_clusters]
          log_prob = torch.logsumexp(ll_stack + log_w, dim=1, keepdim=True)

          return log_prob

      def sample(self, n_samples):
          """Sample from mixture by first selecting cluster, then sampling from it."""
          # 1. Sample cluster indices based on weights
          cluster_indices = torch.multinomial(
              self.weights, n_samples, replacement=True
          )

          # 2. Sample from selected clusters
          samples_list = []
          for h in range(len(self.cluster_spns)):
              mask = (cluster_indices == h)
              n_h = mask.sum().item()
              if n_h > 0:
                  samples_h = self.cluster_spns[h].sample(n_h)
                  samples_list.append((mask, samples_h))

          # 3. Combine samples
          samples = torch.zeros(n_samples, self.num_features, device=self.device)
          for mask, samples_h in samples_list:
              samples[mask] = samples_h

          return samples

      def get_size_bytes(self):
          """Total size of all cluster SPNs."""
          return sum(spn.get_size_bytes() for spn in self.cluster_spns)

  Justification:
  - Why needed: Seng's implementation builds a local mixture per client (client.py:383-397 _build_cluster_mixture)
  - Current gap: We don't have a class to represent this local mixture structure
  - Impact: Foundation for local clustering - without this, we can't properly aggregate local clusters

  Dependencies: None (foundation class)

  Testing:
  # Unit test
  def test_local_cluster_mixture():
      spn1 = LocalSPNWrapper(num_features=5, device='cpu')
      spn2 = LocalSPNWrapper(num_features=5, device='cpu')

      mix = LocalClusterMixture([spn1, spn2], [0.6, 0.4])

      x = torch.randn(10, 5)
      ll = mix.log_prob(x)

      assert ll.shape == (10, 1)
      assert not torch.isnan(ll).any()

  ---
  Phase 2: Implement NaN Marginalization in LocalSPNWrapper

  File: causallearn/utils/FedPC.py

  Location: Modify LocalSPNWrapper.log_prob() method (lines 270-286)

  Current code:
  def log_prob(self, x):
      # 1. Normalize and Permute
      x_norm = self._normalize(x)
      x_perm = self._permute(x_norm)

      # 2. Forward pass through Einet
      ll = self.model(x_perm)

      # 3. Log-Jacobian Correction
      if self.std is not None:
          mask = (~torch.isnan(x)).float()
          log_sigma = torch.log(self.std + 1e-6)
          log_det_jacobian = -(log_sigma * mask).sum(dim=1, keepdim=True)
          ll = ll + log_det_jacobian
      return ll

  Replace with:
  def log_prob(self, x):
      """
      Compute log P(x) with support for marginalization over missing features (NaN).

      Mathematical Foundation (from user insight):
      - For missing features: ∫ P(X_miss) dX_miss = 1
      - In log-space: log(1) = 0
      - Strategy: Fill NaN with zeros (arbitrary), apply observation mask in Jacobian

      This simulates marginalization by excluding missing features from the sum.

      Args:
          x: torch.Tensor [batch, features] - May contain NaN for missing features

      Returns:
          ll: torch.Tensor [batch, 1] - Log-likelihood
      """
      # Detect NaN mask BEFORE any transformation
      nan_mask = torch.isnan(x)  # [batch, features]
      has_nan = nan_mask.any()

      if not has_nan:
          # Fast path: No missing values (standard case)
          x_norm = self._normalize(x)
          x_perm = self._permute(x_norm)
          ll = self.model(x_perm)

          # Jacobian correction for all features
          if self.std is not None:
              log_sigma = torch.log(self.std + 1e-6)
              log_det_jacobian = -log_sigma.sum()  # Scalar, broadcast to batch
              ll = ll + log_det_jacobian

          return ll

      else:
          # Marginalization path: NaN values present
          # Step 1: Fill NaN with zeros (Einet requires valid inputs)
          # The specific fill value doesn't matter because we mask in Jacobian
          x_filled = x.clone()
          x_filled[nan_mask] = 0.0

          # Step 2: Normalize and permute filled data
          x_norm = self._normalize(x_filled)
          x_perm = self._permute(x_norm)

          # Step 3: Forward pass (Einet processes filled data)
          ll = self.model(x_perm)

          # Step 4: CRITICAL - Apply Jacobian correction ONLY to observed features
          if self.std is not None:
              # Observation mask: 1 for observed, 0 for missing
              obs_mask = (~nan_mask).float()  # [batch, features]

              # Log-det-jacobian contribution ONLY from observed features
              # This is the key: missing features contribute 0 (log(1)=0)
              log_sigma = torch.log(self.std + 1e-6)  # [features]
              log_det_jacobian = -(log_sigma * obs_mask).sum(dim=1, keepdim=True)
              ll = ll + log_det_jacobian

          return ll

  Justification:
  - Why needed: CI tests require marginal likelihood P(X_obs) by marginalizing over missing features (cit.py:748-773)
  - Current issue: Einet gets NaN → undefined behavior → CI tests fail → F1=0.000 in hybrid mode
  - Mathematical correctness: Filling NaN + masking Jacobian = simulating ∫ P(X_miss) dX_miss = 1 → log(1) = 0
  - Reference: User's insight about Gaussian integration

  Dependencies: None

  Testing:
  def test_nan_marginalization():
      spn = LocalSPNWrapper(num_features=5, device='cpu')
      spn.train_local(torch.randn(100, 5), epochs=10)

      # Test with NaN
      x = torch.randn(10, 5)
      x[:, 2] = float('nan')  # Feature 2 is missing

      ll = spn.log_prob(x)

      assert ll.shape == (10, 1)
      assert not torch.isnan(ll).any(), "log_prob should not return NaN"
      assert torch.isfinite(ll).all(), "log_prob should be finite"

  ---
  Phase 3: Propagate NaN Handling to GroupMixture

  File: causallearn/utils/FedPC.py

  Location: Modify GroupMixture.log_prob() method (lines 719-752)

  Current code:
  def log_prob(self, x):
      # Extract features for this group
      x_g = x[:, self.feature_indices]

      # Compute log-prob from each client
      client_lls = [spn.log_prob(x_g) for spn in self.client_spns]
      ll_stack = torch.cat(client_lls, dim=1)  # [batch, K_clients]

      # Weighted mixture via logsumexp
      log_weights = torch.log(self.weights + 1e-9).unsqueeze(0)
      log_prob = torch.logsumexp(ll_stack + log_weights, dim=1, keepdim=True)

      return log_prob

  Replace with:
  def log_prob(self, x):
      """
      Compute log P(x_g) = log(Σ_k w_k × P_k(x_g)) for feature group g.

      Now supports NaN values - simply extracts features and delegates to client SPNs.
      Client SPNs (LocalSPNWrapper or LocalClusterMixture) handle NaN internally.

      Args:
          x: torch.Tensor [batch, d_total] - Full feature vector (may contain NaN)

      Returns:
          log_prob: torch.Tensor [batch, 1] - Log-likelihood for this feature group
      """
      # Extract features for this group (may contain NaN if features are missing)
      x_g = x[:, self.feature_indices]

      # Compute log-prob from each client SPN
      # Each client SPN handles NaN internally via Phase 2 fix
      client_lls = [spn.log_prob(x_g) for spn in self.client_spns]
      ll_stack = torch.cat(client_lls, dim=1)  # [batch, K_clients]

      # Weighted mixture via logsumexp (numerically stable)
      log_weights = torch.log(self.weights + 1e-9).unsqueeze(0)
      log_prob = torch.logsumexp(ll_stack + log_weights, dim=1, keepdim=True)

      return log_prob

  Justification:
  - Why needed: Propagate NaN handling from LocalSPNWrapper up through the hierarchy
  - Current state: Already mostly correct - just needs to delegate to client SPNs
  - Key insight: No special NaN handling needed here because LocalSPNWrapper already handles it (Phase 2)
  - Impact: Ensures GroupMixture works with marginalized queries

  Dependencies: Phase 2 (LocalSPNWrapper NaN handling)

  Testing:
  def test_group_mixture_nan():
      spn1 = LocalSPNWrapper(num_features=3, device='cpu')
      spn2 = LocalSPNWrapper(num_features=3, device='cpu')

      mix = GroupMixture([spn1, spn2], [0.5, 0.5], feature_indices=[0,1,2])

      # Full data with NaN in feature subset
      x = torch.randn(10, 5)
      x[:, 1] = float('nan')  # Feature 1 is missing

      ll = mix.log_prob(x)

      assert not torch.isnan(ll).any()

  ---
  Phase 4: Implement Local Clustering in FedCDH

  File: causallearn/search/FCMBased/FedCDH/FedCDH.py

  Location: Replace clustering + training logic (lines 396-634)

  Current structure (WRONG - Global clustering):
  # Lines 396-465: BIC-based global cluster selection
  for h_candidate in range(2, 6):
      fed_km = SimulatedFederatedKMeans(n_clusters=h_candidate)
      fed_km.fit(X_splits, feature_maps, self.scenario)
      # ... BIC calculation ...

  # Lines 539-634: Train SPNs per (global_cluster, client) pair
  for h in range(num_clusters):  # Global cluster first
      for k in range(K_clients):  # Client second
          local_data_h = X_splits[k][labels_splits[k] == h]  # FRAGMENTED!
          if len(local_data_h) > 2:
              leaf = LocalSPNWrapper(...)
              leaf.train_local(local_data_h)

  Replace with (CORRECT - Local clustering):

  # Lines 396-800: Local clustering and training
  # Following Seng et al. (2025) client.py:343-397

  logging.info(f"[V2 LOCAL CLUSTERING] Following Seng et al. (2025) Algorithm 1")
  logging.info(f"  Clustering scope: LOCAL per client (not global)")
  logging.info(f"  num_clusters: {num_clusters} local clusters per client")

  # Storage for client models
  client_local_mixtures = []

  # CRITICAL CHANGE: Loop over CLIENTS first, then cluster locally
  for k in range(self.K_clients):
      logging.info(f"\n[Client {k}] Starting local clustering and training")

      # Get this client's data
      if self.scenario == "vertical":
          # Vertical: client_data already contains only their features + context
          client_data = X_splits[k]
      else:
          # Horizontal/Hybrid: client_data is their samples of full feature set
          client_data = X_splits[k]

      n_client = client_data.shape[0]
      local_d = client_data.shape[1]

      logging.info(f"[Client {k}] Data shape: {client_data.shape} (n={n_client}, d={local_d})")

      # Decide whether to use clustering
      # Use clustering if: num_clusters > 1 AND sufficient data
      use_local_clustering = (num_clusters > 1) and (n_client >= num_clusters * 20)

      if use_local_clustering:
          logging.info(f"[Client {k}] Performing LOCAL K-means with K={num_clusters}")

          # LOCAL K-means: cluster only this client's data
          from sklearn.cluster import KMeans

          # macOS fix: force single-threaded to avoid OpenMP hang
          old_omp = os.environ.get('OMP_NUM_THREADS', None)
          os.environ['OMP_NUM_THREADS'] = '1'

          try:
              kmeans = KMeans(
                  n_clusters=num_clusters,
                  random_state=42 + k,  # Different seed per client
                  n_init=10,
                  max_iter=100
              )
              local_cluster_labels = kmeans.fit_predict(client_data)
          finally:
              # Restore original OMP setting
              if old_omp is not None:
                  os.environ['OMP_NUM_THREADS'] = old_omp
              else:
                  os.environ.pop('OMP_NUM_THREADS', None)

          # Check cluster distribution
          unique_clusters, cluster_counts = np.unique(local_cluster_labels, return_counts=True)
          logging.info(f"[Client {k}] Cluster distribution: {dict(zip(unique_clusters, cluster_counts))}")

          # Train K SPNs for K local clusters
          cluster_spns = []
          cluster_weights = []

          for h in range(num_clusters):
              mask = (local_cluster_labels == h)
              cluster_size = mask.sum()

              if cluster_size < 10:
                  logging.warning(f"[Client {k}] Cluster {h} has only {cluster_size} samples, skipping")
                  continue

              cluster_data = client_data[mask]

              logging.info(f"[Client {k}] Training SPN for local cluster {h}: {cluster_size} samples")

              # Train SPN on this local cluster
              spn_kh = LocalSPNWrapper(
                  num_features=local_d,
                  device=self.device,
                  depth=depth,
                  num_sums=num_sums,
                  num_leaves=num_leaves,
                  num_repetitions=num_repetitions,
                  seed=10000 + 1000 * k + 100 * h + i
              )

              spn_kh.train_local(
                  cluster_data,
                  epochs=epochs,
                  lr=lr,
                  l1_weight=l1_weight,
                  l2_weight=l2_weight,
                  dropout=dropout
              )

              cluster_spns.append(spn_kh)
              cluster_weights.append(cluster_size)

          if len(cluster_spns) == 0:
              logging.error(f"[Client {k}] No valid clusters found, falling back to single SPN")
              use_local_clustering = False  # Fall back to single SPN
          else:
              # Normalize cluster weights to sum to 1
              cluster_weights = np.array(cluster_weights, dtype=np.float32) / n_client

              logging.info(f"[Client {k}] Built local mixture: {len(cluster_spns)} SPNs, weights={cluster_weights}")

              # Build LOCAL mixture: P_k(X) = Σ_h w_{k,h} × SPN_{k,h}(X)
              local_mixture = LocalClusterMixture(
                  cluster_spns=cluster_spns,
                  cluster_weights=cluster_weights,
                  device=self.device
              )

              client_local_mixtures.append(local_mixture)

      if not use_local_clustering:
          # No clustering: single SPN for this client
          logging.info(f"[Client {k}] Training single SPN (no clustering)")

          single_spn = LocalSPNWrapper(
              num_features=local_d,
              device=self.device,
              depth=depth,
              num_sums=num_sums,
              num_leaves=num_leaves,
              num_repetitions=num_repetitions,
              seed=10000 + 1000 * k + i
          )

          single_spn.train_local(
              client_data,
              epochs=epochs,
              lr=lr,
              l1_weight=l1_weight,
              l2_weight=l2_weight,
              dropout=dropout
          )

          # Wrap single SPN as a "mixture" with one component (for consistency)
          local_mixture = LocalClusterMixture(
              cluster_spns=[single_spn],
              cluster_weights=np.array([1.0], dtype=np.float32),
              device=self.device
          )

          client_local_mixtures.append(local_mixture)

      # Store for evaluation
      self.local_spns.append(local_mixture)

  logging.info(f"\n[V2 LOCAL CLUSTERING] Complete: {len(client_local_mixtures)} client models trained")

  # Continue with scenario-specific global aggregation...
  # (Existing code for horizontal/vertical/hybrid aggregation remains the same)

  Justification:
  - Why needed: Current global clustering fragments data (1200 samples → 9 SPNs with some having <25 samples)
  - Seng's approach: Local clustering preserves data (400 samples/client → 2 SPNs with 200 samples each)
  - Impact: Fixes F1=0.000 failures across ALL modes (horizontal/vertical/hybrid)
  - Evidence: SPN_STRUCTURE_COMPARISON.md shows this is the CRITICAL bug

  Key changes:
  1. Loop order: Client first (k), then cluster (h) — NOT cluster first!
  2. K-means scope: kmeans.fit_predict(client_data) — NOT fit_predict(all_data)
  3. Data per SPN: 400/2=200 samples — NOT 1200/9=133 samples
  4. New class used: LocalClusterMixture wraps the K local cluster SPNs

  Dependencies: Phase 1 (LocalClusterMixture class)

  Testing:
  def test_local_clustering_preserves_data():
      # After implementation, verify data distribution
      # Each client with 400 samples, K=2 clusters
      # Should have: [~200, ~200] per client (not [22, 151, 227])

      for k in range(K_clients):
          assert all(size >= 150 for size in cluster_sizes[k]), \
              f"Client {k} has insufficient cluster sizes: {cluster_sizes[k]}"

  ---
  Phase 5: Update Global Aggregation to Use LocalClusterMixture

  File: causallearn/search/FCMBased/FedCDH/FedCDH.py

  Location: Lines 650-920 (scenario-specific aggregation)

  Current code expects: List of LocalSPNWrapper objects per client

  After Phase 4: We have client_local_mixtures — List of LocalClusterMixture objects

  Changes needed:

  5A: Horizontal Mode (Lines 650-700)

  Current:
  if self.scenario == "horizontal":
      fed_spn = GlobalFedSPN(
          components=client_spns,  # OLD: List of LocalSPNWrapper
          weights=client_weights,
          device=self.device,
          strategy="mixture"
      )

  Change to:
  if self.scenario == "horizontal":
      # client_local_mixtures: List[LocalClusterMixture]
      # Each is already a mixture over local clusters
      fed_spn = GlobalFedSPN(
          components=client_local_mixtures,  # NEW: List of LocalClusterMixture
          weights=client_weights,
          device=self.device,
          strategy="mixture"
      )

  Justification: GlobalFedSPN is agnostic to component type — just needs .log_prob() method

  5B: Vertical Mode (Lines 702-803)

  No changes needed — vertical mode will use the same local clustering logic from Phase 4

  Justification: Vertical already works per-client, just needs local mixtures instead of single SPNs

  5C: Hybrid Mode (Lines 805-920)

  Critical section: Feature-subspace training

  Current approach: Trains NEW SPNs per feature subspace (from DAY2_HYBRID_FIX)

  After Phase 4: We have client_local_mixtures available

  Key decision: Should hybrid use:
  - Option A: Re-train SPNs per feature subspace (current approach after Fix #1)
  - Option B: Reuse local mixtures and extract feature subspaces

  Recommended: Option A (keep current hybrid fix, just use local clustering base)

  Justification:
  - Hybrid feature groups may not align with full client feature sets
  - Better to train specialized SPNs per feature group
  - Local clustering ensures sufficient data for each feature-group SPN

  Code (already implemented in hybrid fix, just verify compatibility):
  elif self.scenario == "hybrid":
      # Note: Hybrid trains NEW SPNs per feature subspace
      # But benefits from local clustering ensuring sufficient base data

      # Current feature-subspace training logic is correct
      # Just ensure it uses the same local clustering parameters
      # (This section doesn't need changes - already fixed in DAY2_HYBRID_FIX)

  ---
  Phase 6: Validation and Testing

  File: tests/test/test_local_clustering.py (NEW)

  Create comprehensive validation tests:

  import numpy as np
  import torch
  from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH
  from causallearn.utils.FedPC import LocalClusterMixture, LocalSPNWrapper

  def test_local_cluster_mixture_basic():
      """Test LocalClusterMixture creation and forward pass."""
      spn1 = LocalSPNWrapper(num_features=5, device='cpu', seed=42)
      spn2 = LocalSPNWrapper(num_features=5, device='cpu', seed=43)

      # Train on dummy data
      data1 = torch.randn(100, 5)
      data2 = torch.randn(100, 5)
      spn1.train_local(data1, epochs=10)
      spn2.train_local(data2, epochs=10)

      # Create mixture
      mix = LocalClusterMixture([spn1, spn2], [0.6, 0.4])

      # Test log_prob
      x = torch.randn(20, 5)
      ll = mix.log_prob(x)

      assert ll.shape == (20, 1)
      assert not torch.isnan(ll).any()
      assert torch.isfinite(ll).all()

      # Test sample
      samples = mix.sample(50)
      assert samples.shape == (50, 5)
      print("✓ LocalClusterMixture basic test passed")


  def test_nan_marginalization():
      """Test NaN handling in LocalSPNWrapper."""
      spn = LocalSPNWrapper(num_features=5, device='cpu', seed=42)

      # Train
      data = torch.randn(100, 5)
      spn.train_local(data, epochs=10)

      # Test with NaN
      x = torch.randn(20, 5)
      x[:, 2] = float('nan')  # Feature 2 is missing
      x[:, 4] = float('nan')  # Feature 4 is missing

      ll = spn.log_prob(x)

      assert ll.shape == (20, 1)
      assert not torch.isnan(ll).any(), "log_prob returned NaN!"
      assert torch.isfinite(ll).all(), "log_prob not finite!"
      print("✓ NaN marginalization test passed")


  def test_local_vs_global_clustering():
      """Compare data distribution: local vs global clustering."""
      from sklearn.cluster import KMeans

      # Simulate 3 clients with 400 samples each
      K_clients = 3
      n_per_client = 400
      d = 10
      num_clusters = 3

      X_splits = [np.random.randn(n_per_client, d) for _ in range(K_clients)]

      # GLOBAL clustering (old approach - WRONG)
      X_all = np.vstack(X_splits)
      global_kmeans = KMeans(n_clusters=num_clusters, random_state=42)
      global_labels = global_kmeans.fit_predict(X_all)

      global_distribution = []
      curr = 0
      for k in range(K_clients):
          client_labels = global_labels[curr:curr+n_per_client]
          for h in range(num_clusters):
              size = (client_labels == h).sum()
              global_distribution.append(size)
          curr += n_per_client

      # LOCAL clustering (new approach - CORRECT)
      local_distribution = []
      for k in range(K_clients):
          local_kmeans = KMeans(n_clusters=num_clusters, random_state=42+k)
          local_labels = local_kmeans.fit_predict(X_splits[k])
          for h in range(num_clusters):
              size = (local_labels == h).sum()
              local_distribution.append(size)

      print(f"\nGlobal clustering distribution (9 SPNs): {global_distribution}")
      print(f"  Min: {min(global_distribution)}, Max: {max(global_distribution)}")
      print(f"  Problem: Some SPNs have < 50 samples")

      print(f"\nLocal clustering distribution (9 SPNs): {local_distribution}")
      print(f"  Min: {min(local_distribution)}, Max: {max(local_distribution)}")
      print(f"  All SPNs have sufficient data!")

      # Validation: Local clustering should give more balanced distribution
      assert min(local_distribution) > min(global_distribution), \
          "Local clustering should avoid extreme fragmentation"

      # Validation: All local clusters should have reasonable size
      assert all(size >= 80 for size in local_distribution), \
          f"Local clustering failed: {local_distribution}"

      print("✓ Local vs global clustering comparison passed")


  if __name__ == "__main__":
      test_local_cluster_mixture_basic()
      test_nan_marginalization()
      test_local_vs_global_clustering()
      print("\n✅ All validation tests passed!")

  Run tests:
  python tests/test/test_local_clustering.py

  ---
  Phase 7: End-to-End Smoke Test

  File: Run existing benchmark with fixes

  Quick smoke test (CPU, fast):
  python tests/test/test_fedcdh_benchmark.py \
    --config quick \
    --data-type linear \
    --device cpu \
    --seeds 42 \
    --num-local-clusters 2 \
    --skip-eval

  Expected results BEFORE fixes:
  Horizontal: F1=0.000 (data fragmentation)
  Vertical:   F1=0.222 (data fragmentation)
  Hybrid:     F1=0.000 (fragmentation + NaN issue)

  Expected results AFTER fixes:
  Horizontal: F1=0.5-0.7 (local clustering fixes fragmentation)
  Vertical:   F1=0.6-0.8 (local clustering + better data)
  Hybrid:     F1=0.3-0.5 (local clustering + NaN marginalization working)

  Full validation (CPU, with evaluation):
  python tests/test/test_fedcdh_benchmark.py \
    --config quick \
    --data-type linear \
    --device cpu \
    --seeds 42 \
    --num-local-clusters 2

  Expected:
  - No NaN errors in logs
  - Positive train log-likelihoods
  - All 3 modes complete successfully
  - F1 scores > 0.3 for all modes

  ---
  Implementation Timeline & Dependencies

  Day 1: Foundation
  ├─ Morning:   Phase 1 (LocalClusterMixture class) [2 hours]
  ├─ Afternoon: Phase 2 (NaN marginalization in LocalSPNWrapper) [2 hours]
  └─ Evening:   Phase 3 (NaN in GroupMixture) [1 hour]
             └─ Unit tests for Phases 1-3 [1 hour]

  Day 2: Core Fix
  ├─ Morning:   Phase 4 (Local clustering in FedCDH) [4 hours] ← CRITICAL
  ├─ Afternoon: Phase 5 (Update global aggregation) [2 hours]
  └─ Evening:   Phase 6 (Validation tests) [2 hours]

  Day 3: Testing & Validation
  ├─ Morning:   Phase 7 (Smoke tests) [2 hours]
  ├─ Afternoon: GPU quick test (medium config) [2 hours runtime]
  └─ Evening:   Analysis and documentation [2 hours]

  Total: 2.5 days implementation + validation

  Critical Path:
  Phase 1 (LocalClusterMixture)
      ↓
  Phase 4 (Local clustering) ← MOST CRITICAL
      ↓
  Phase 5 (Global aggregation updates)
      ↓
  Phase 7 (End-to-end testing)

  Parallel track:
  Phase 2 (NaN in LocalSPNWrapper)
      ↓
  Phase 3 (NaN in GroupMixture)
      ↓
  Phase 7 (Hybrid mode testing)

  ---
  Justification Summary

  Why Phase 1 (LocalClusterMixture)?

  - Seng's implementation has this concept (client.py:383-397)
  - Needed to represent P_k(X) = Σ_h w_{k,h} × SPN_{k,h}(X)
  - Foundation for local clustering

  Why Phase 2 (NaN Marginalization)?

  - CI tests require P(X_obs) by marginalizing P(X_obs, X_miss)
  - User insight: ∫ P(X_miss) dX_miss = 1 → log(1) = 0
  - Hybrid mode completely broken without this (F1=0.000)

  Why Phase 3 (GroupMixture NaN)?

  - Propagates NaN handling through hierarchy
  - Minimal change (delegates to LocalSPNWrapper)

  Why Phase 4 (Local Clustering)?

  - THE CRITICAL FIX - Fixes data fragmentation
  - Global clustering: 1200/(3×3) = 133 avg, some < 25 samples
  - Local clustering: 400/2 = 200 samples per SPN
  - Impacts ALL modes (H/V/Hy)

  Why Phase 5 (Global Aggregation)?

  - Update references to use LocalClusterMixture instead of LocalSPNWrapper
  - Minimal changes (mostly type updates)

  Why Phases 6-7 (Testing)?

  - Validate each component independently
  - End-to-end verification
  - Compare against expected improvements

  ---
  Success Metrics

  After implementation, we should achieve:
  ┌────────────┬─────────────┬──────────┬──────────┬────────┐
  │    Mode    │   Metric    │  Before  │  After   │ Target │
  ├────────────┼─────────────┼──────────┼──────────┼────────┤
  │ Horizontal │ Skeleton F1 │ 0.000    │ 0.5-0.7  │ >0.5 ✓ │
  ├────────────┼─────────────┼──────────┼──────────┼────────┤
  │ Horizontal │ Data/SPN    │ 14-227   │ 190-210  │ >150 ✓ │
  ├────────────┼─────────────┼──────────┼──────────┼────────┤
  │ Vertical   │ Skeleton F1 │ 0.222    │ 0.6-0.8  │ >0.6 ✓ │
  ├────────────┼─────────────┼──────────┼──────────┼────────┤
  │ Hybrid     │ Skeleton F1 │ 0.000    │ 0.3-0.5  │ >0.3 ✓ │
  ├────────────┼─────────────┼──────────┼──────────┼────────┤
  │ Hybrid     │ NaN errors  │ Many     │ Zero     │ Zero ✓ │
  ├────────────┼─────────────┼──────────┼──────────┼────────┤
  │ All        │ Train LL    │ Negative │ Positive │ >0 ✓   │
  └────────────┴─────────────┴──────────┴──────────┴────────┘
  Key validation points:
  1. ✅ No data fragmentation (all SPNs have >150 samples)
  2. ✅ No NaN in log-likelihood outputs
  3. ✅ All modes complete without errors
  4. ✅ F1 scores > 0 for all modes
  5. ✅ Positive train log-likelihoods

  ---
  This roadmap provides a clear, step-by-step implementation plan grounded in:
  - Seng et al. (2025) reference implementation
  - Your mathematical insight about marginalization
  - Documented analysis (SPN_STRUCTURE_COMPARISON.md)
  - Current codebase structure

  Each phase has clear justification, dependencies, and validation criteria.
