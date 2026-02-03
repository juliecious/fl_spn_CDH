import torch
import torch.nn as nn
import numpy as np
import logging
from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.normal import Normal
from sklearn.cluster import KMeans


class LocalLeafSPN(nn.Module):
    def __init__(
        self,
        num_features,
        device="cpu",
        depth=2,
        num_sums=20,
        num_leaves=20,
        num_repetitions=10,
    ):
        super().__init__()
        self.device = device

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

    def train_local(self, data, epochs=50, lr=0.005):
        """Train this specific leaf on its data slice."""
        if len(data) < 5:
            return 0.0

        self.model.train()

        # [FIX 2] Optimization Stability
        # Adam is fine if LR is low enough. 0.01 was too high.
        # Alternatively use SGD(lr=0.001) like Jonas, but Adam 0.005 is a good middle ground.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        data_t = torch.tensor(data, dtype=torch.float32).to(self.device)

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
        return self.model.sample(n)

    def log_prob(self, x):
        return self.model(x)


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
        # Grandchildren are LocalLeafSPNs.

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
            leaf = LocalLeafSPN(
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
