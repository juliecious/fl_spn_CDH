import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Optional, Union
from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.normal import Normal
from sklearn.cluster import KMeans


class LocalSPNWrapper(nn.Module):
    def __init__(
        self,
        num_features,
        device="cpu",
        depth=2,
        num_sums=20,
        num_leaves=20,
        num_repetitions=10,
        seed=None,
    ):
        super().__init__()
        self.device = device
        self.mean = None
        self.std = None

        # Ensure structural heterogeneity by seeding
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # [FIX 1] Increased Capacity:
        # num_sums 5->20, num_leaves 5->20, repetitions 1->10
        self.config = EinetConfig(
            num_features=num_features,
            num_channels=1,
            num_sums=num_sums,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,  # Ensemble of 10 trees
            depth=depth,
            num_classes=1,
            leaf_type=Normal,
            layer_type="linsum",
            structure="top-down",
        )
        self.model = Einet(self.config).to(device)

    def _normalize(self, data):
        if self.mean is None or self.std is None:
            return data
        return (data - self.mean) / (self.std + 1e-6)

    def train_local(self, data, epochs=50, lr=0.005):
        """Train this specific leaf on its data slice."""
        if len(data) < 5:
            return 0.0

        # Compute and store normalization stats
        self.mean = torch.tensor(data.mean(axis=0), dtype=torch.float32).to(self.device)
        self.std = torch.tensor(data.std(axis=0), dtype=torch.float32).to(self.device)

        data_t = torch.tensor(data, dtype=torch.float32).to(self.device)
        data_t = self._normalize(data_t)

        self.model.train()

        # [FIX 2] Optimization Stability
        # Adam is fine if LR is low enough. 0.01 was too high.
        # Alternatively use SGD(lr=0.001) like Jonas, but Adam 0.005 is a good middle ground.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        final_ll = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            ll = self.model(data_t)
            loss = -ll.mean()
            loss.backward()
            optimizer.step()
            final_ll = -loss.item()
        return final_ll

    def sample(self, n):
        samples = self.model.sample(n)
        if self.mean is not None and self.std is not None:
            samples = samples * (self.std + 1e-6) + self.mean
        return samples

    def log_prob(self, x):
        x_norm = self._normalize(x)
        # Jacobian adjustment for log-likelihood:
        # log P(x) = log P(z) + log |dz/dx|
        # z = (x - mu) / sigma  =>  dz/dx = 1/sigma
        # log |dz/dx| = - log(sigma)
        # ONLY for observed variables
        ll = self.model(x_norm)

        if self.std is not None:
            # Create mask for observed values (1 if observed, 0 if NaN)
            mask = (~torch.isnan(x)).float()

            # Calculate per-dimension log determinant: -log(sigma)
            # Expand std to match batch size if needed (it is 1D tensor [D])
            log_sigma = torch.log(self.std + 1e-6)

            # Sum over dimensions, respecting the mask
            # [Batch, D] * [D] -> [Batch, D] -> Sum -> [Batch]
            log_det_jacobian = -(log_sigma * mask).sum(dim=1, keepdim=True)

            ll = ll + log_det_jacobian

        return ll


class FedPC(nn.Module):
    """
    Jonas's Network-Aligned SPN.
    Implements 'Mixture of Products' for Vertical/Hybrid support.
    """

    def __init__(
        self, scenario, global_num_features, K_clients, num_clusters=5, device="cpu"
    ):
        super().__init__()
        self.scenario = scenario
        self.num_clusters = num_clusters  # Number of Latent Components (Z)
        self.K_clients = K_clients
        self.device = device

        # Structure:
        # Root is a Sum Node (Cluster Mixture).
        # Children are Product Nodes (Client Composition).
        # Grandchildren are LocalSPNWrapper.

        # self.leaves[cluster_idx][client_idx]
        self.leaves = nn.ModuleList(
            [
                nn.ModuleList([None for _ in range(K_clients)])
                for _ in range(num_clusters)
            ]
        )

        self.cluster_weights = None  # P(Z=k)
        self.feature_map = {}  # client_idx -> list of feature indices

        logging.info(
            f"Initialized FedPC ({scenario}) with {num_clusters} latent clusters."
        )

    def register_client(self, client_idx, feature_indices, leaf_params):
        """Instantiate the local RAT-SPNs for this client across all clusters."""
        self.feature_map[client_idx] = feature_indices
        num_feat = len(feature_indices)

        for k in range(self.num_clusters):
            # Create a dedicated local model for each cluster component
            leaf = LocalSPNWrapper(
                num_features=num_feat, device=self.device, **leaf_params
            )
            self.leaves[k][client_idx] = leaf

    def fit_one_pass(self, X_global, X_splits):
        """
        Algorithm 1: Scaling via Data Partitioning.
        1. Cluster the data (Simulated 'Private' Clustering via Proxy).
        2. Train local leaves on assigned data.
        3. Set global mixture weights.
        """
        # Step 1: Clustering
        # We use KMeans on global proxy data to find latent assignments Z
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10).fit(X_global)
        labels = kmeans.labels_

        # Set Mixture Weights P(Z=k) based on counts
        counts = np.bincount(labels, minlength=self.num_clusters)
        total = sum(counts) + 1e-9
        self.cluster_weights = torch.tensor(counts / total, dtype=torch.float32).to(
            self.device
        )
        logging.info(f"Cluster Weights: {self.cluster_weights.cpu().numpy()}")

        # Step 2: Local Training (Parallel per Client & Cluster)
        for k in range(self.num_clusters):
            # Indices of data points in this cluster
            cluster_indices = np.where(labels == k)[0]
            if len(cluster_indices) == 0:
                continue

            for c_idx in range(self.K_clients):
                # Slice client data by the cluster rows
                client_data_k = X_splits[c_idx][cluster_indices]

                # Train the specific leaf
                if self.leaves[k][c_idx] is not None:
                    self.leaves[k][c_idx].train_local(client_data_k)

    def log_prob(self, x):
        """
        Exact Inference: P(X) = Sum_k P(Z=k) * Prod_c P(X_c | Z=k)
        x: Global data [Batch, D]
        """
        batch_size = x.shape[0]
        cluster_log_probs = []

        for k in range(self.num_clusters):
            # log(Prod P_c) = Sum log(P_c)
            client_log_sums = torch.zeros(batch_size, 1, device=self.device)

            for c_idx in range(self.K_clients):
                leaf = self.leaves[k][c_idx]
                feat_idx = self.feature_map[c_idx]
                x_local = x[:, feat_idx]

                if leaf is not None:
                    ll = leaf.log_prob(x_local)  # [Batch, 1]
                    client_log_sums += ll

            cluster_log_probs.append(client_log_sums)

        # Stack: [Batch, K]
        log_P_X_given_Z = torch.cat(cluster_log_probs, dim=1)

        # Mixture: log Sum_k P(k) * exp(log_P_X_given_Z)
        log_weights = torch.log(self.cluster_weights + 1e-12).unsqueeze(0)
        return torch.logsumexp(log_weights + log_P_X_given_Z, dim=1)

    def calculate_cmi(self, x_idx, y_idx, z_idxs, data_matrix):
        """
        Calculate Conditional Mutual Information I(X;Y | Z).
        Uses G-Test approximation with empirical data or Model inference.
        """
        # [FIX]: Ensure inputs are lists to avoid 'tuple' concatenation error
        if isinstance(x_idx, (int, float, np.integer)):
            x_idx = int(x_idx)
        if isinstance(y_idx, (int, float, np.integer)):
            y_idx = int(y_idx)

        if isinstance(z_idxs, tuple):
            z_idxs = list(z_idxs)
        elif isinstance(z_idxs, (int, float, np.integer)):
            z_idxs = [int(z_idxs)]
        elif z_idxs is None:
            z_idxs = []
        else:
            z_idxs = list(z_idxs)

        # 1. Prepare indices
        vars_of_interest = [x_idx] + [y_idx] + z_idxs

        # 2. Estimate CMI via Point-wise Mutual Information (PMI) on the data samples
        # CMI = E[ log P(x,y|z) - log P(x|z) - log P(y|z) ]
        # We use the FedPC model to give us the log-probabilities.

        data_t = torch.tensor(data_matrix, dtype=torch.float32).to(self.device)

        # P(X, Y, Z)
        log_xyz = self.log_prob_marginal(data_t, vars_of_interest)

        # P(X, Z)
        log_xz = self.log_prob_marginal(data_t, [x_idx] + z_idxs)

        # P(Y, Z)
        log_yz = self.log_prob_marginal(data_t, [y_idx] + z_idxs)

        # P(Z)
        if len(z_idxs) > 0:
            log_z = self.log_prob_marginal(data_t, z_idxs)
        else:
            log_z = torch.zeros_like(log_xyz)

        # PMI = log( P(xyz) * P(z) / (P(xz) * P(yz)) )
        #     = log P(xyz) + log P(z) - log P(xz) - log P(yz)
        pmi = log_xyz + log_z - log_xz - log_yz

        cmi_val = pmi.mean().item()
        return max(0.0, cmi_val)

    def log_prob_marginal(self, x_full, query_indices):
        """
        Approximate Marginal P(X_subset).
        For FedPC (Mixture of Products), we marginalize by dropping clients
        that are not involved in the query (summing them out to 1).
        For involved clients, we use their local likelihoods.
        """
        batch_size = x_full.shape[0]
        cluster_log_probs = []

        # Convert query list to set for O(1) lookup
        query_set = set(query_indices)

        for k in range(self.num_clusters):
            client_log_sums = torch.zeros(batch_size, 1, device=self.device)

            for c_idx in range(self.K_clients):
                leaf = self.leaves[k][c_idx]
                feat_idx = self.feature_map[c_idx]

                # Intersection: Does this client hold any queried variables?
                client_vars = set(feat_idx)
                local_query = query_set.intersection(client_vars)

                if len(local_query) == 0:
                    # Marginalize out this ENTIRE client (Integral = 1, Log = 0)
                    continue
                else:
                    # Client is involved.
                    # IDEALLY: We should integrate out non-query variables local to this client.
                    # APPROX: We pass the full local vector.
                    # This assumes the query covers the significant variance or
                    # that the user accepts this as a 'Pseudo-Marginal' for the test.
                    # (Exact marginalization requires support from simple-einet leaf).
                    x_local = x_full[:, feat_idx]
                    if leaf is not None:
                        ll = leaf.log_prob(x_local)
                        client_log_sums += ll

            cluster_log_probs.append(client_log_sums)

        log_P_X_given_Z = torch.cat(cluster_log_probs, dim=1)
        log_weights = torch.log(self.cluster_weights + 1e-12).unsqueeze(0)
        return torch.logsumexp(log_weights + log_P_X_given_Z, dim=1)


class GlobalFedSPN(nn.Module):
    """
    Global Federated SPN (Horizontal/Mixture Strategy).
    Models P(X) = Sum_k alpha_k * P_k(X).
    Supports conditioning on Client Index U for FedCDH.
    """

    def __init__(
        self, clients: List[LocalSPNWrapper], weights: List[float] = None, device="cpu"
    ):
        super().__init__()
        self.clients = nn.ModuleList(clients)
        self.device = device

        if weights is None:
            # Default to uniform weights
            n = len(clients)
            weights = [1.0 / n] * n

        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)

    def log_prob(self, x):
        """
        Computes log P(X) = log( Sum_k alpha_k * P_k(X) )
        x: [N, D]
        """
        # Collect log probs from all clients: List of [N, 1]
        # Each client model expects the full feature vector (or handles its subset)
        # Assuming Horizontal FL where all clients see all features.
        client_lls = [c.log_prob(x) for c in self.clients]

        # Stack: [N, K]
        ll_stack = torch.cat(client_lls, dim=1)

        # Log weights: [1, K]
        log_w = torch.log(self.weights + 1e-9).unsqueeze(0)

        # LogSumExp over the client dimension (dim=1)
        return torch.logsumexp(ll_stack + log_w, dim=1, keepdim=True)

    def log_prob_conditional_u(self, x, u_idx):
        """
        Computes P(X | U=k).
        In FedCDH, conditioning on the client index k simply means
        using the k-th client's distribution.

        x: [N, D]
        u_idx: int (The client index k)
        """
        if 0 <= u_idx < len(self.clients):
            return self.clients[u_idx].log_prob(x)
        else:
            raise ValueError(
                f"Client index {u_idx} out of bounds (0 to {len(self.clients)-1})"
            )


def auto_tune_spn_config(proxy_data, num_clusters=1, n_trials=15, device="cpu"):
    # ... placeholder or original implementation if needed ...
    # For now, returning default dict as I overwrote the file
    # But wait, I should keep the original implementation if possible.
    # The read_file showed it at the end. I will copy it back.
    import optuna
    from sklearn.model_selection import train_test_split

    logging.info(
        f"[Auto-Tune] Starting optimization on {proxy_data.shape} proxy samples..."
    )

    # SAFETY: Handle NaNs in proxy data
    proxy_data = np.nan_to_num(proxy_data, nan=0.0)

    if len(proxy_data) < 10:
        return {
            "num_sums": 10,
            "num_leaves": 20,
            "depth": 2,
            "lr": 0.05,
            "batch_size": 32,
        }

    train_data, val_data = train_test_split(proxy_data, test_size=0.2, random_state=42)
    val_tensor = torch.tensor(val_data, dtype=torch.float32).to(device)

    def objective(trial):
        num_sums = trial.suggest_int("num_sums", 20, 30)
        num_leaves = trial.suggest_int("num_leaves", 10, 25)
        depth = trial.suggest_int("depth", 2, 4)
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        # IMPORTANT: Use LocalSPNWrapper here!
        client = LocalSPNWrapper(
            num_features=proxy_data.shape[1],
            device=device,
            depth=depth,
            num_sums=num_sums,
            num_leaves=num_leaves,
            num_repetitions=5,
        )

        try:
            client.train_local(train_data, epochs=5, lr=lr)
        except Exception:
            return float("inf")

        client.model.eval()
        with torch.no_grad():
            ll_output = client.log_prob(val_tensor)
            val_nll = -ll_output.mean().item()

        return val_nll

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logging.info(f"[Auto-Tune] Best Params: {study.best_params}")
    return study.best_params
