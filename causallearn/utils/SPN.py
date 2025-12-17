import numpy as np
import torch
import logging
from typing import List, Dict, Optional, Union
from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.normal import Normal


class FederatedSPNBase:
    """
    Base class containing shared utilities for Federated SPNs.
    """

    def __init__(self, device="cpu"):
        self.device = device


class ClientSPN(FederatedSPNBase):
    """
    Client Node: Trains a local SPN on its private data partition.
    Supports learning K components (clusters) for Vertical/Hybrid scenarios.
    """

    def __init__(
        self,
        client_id: int,
        num_features: int,
        num_clusters: int = 1,  # K in the paper (1 for Horizontal, >1 for Vertical)
        device="cpu",
    ):
        super().__init__(device)
        self.client_id = client_id
        self.num_features = num_features
        self.num_clusters = num_clusters

        # Depth must be <= log2(num_features)
        max_depth = int(np.floor(np.log2(num_features)))
        # Use depth 2 if possible, otherwise use the maximum allowed
        actual_depth = min(2, max_depth)

        # Configuration matches the paper's "Federated PC" setup
        # num_classes = num_clusters (The Latent Variable L)
        self.config = EinetConfig(
            num_features=num_features,
            num_channels=1,
            num_sums=5,
            num_leaves=10,
            num_repetitions=5,
            depth=actual_depth,
            num_classes=num_clusters,  # Crucial: Each class represents a 'cluster' k
            leaf_type=Normal,
            layer_type="linsum",
            structure="top-down",
        )

        self.model = Einet(self.config).to(self.device)
        self.data_count = 0

    def train(self, X_local: np.ndarray, epochs=20, lr=0.01, batch_size=128):
        """
        Trains the local model.
        """
        self.data_count = X_local.shape[0]
        self.model.train()

        # simple-einet expects [N, C, D] or [N, D]
        tensor_data = torch.tensor(X_local, dtype=torch.float32).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Simple training loop
        # (In production, use DataLoader)
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward: Returns [N, K] log-likelihoods per class (cluster)
            ll_output = self.model(tensor_data)

            # For standard training, we maximize Log-Likelihood of data
            # If K=1 (Horizontal), ll_output is [N, 1] -> mean()
            # If K>1 (Vertical), we sum over latent K (LogSumExp) -> mean()
            if self.num_clusters > 1:
                # Log-Likelihood of data = LogSumExp_k( P(X|C=k) + log P(C=k) )
                # simple-einet's forward returns P(X | C) if num_classes > 1 (check einet.py forward)
                # Actually, einet.py forward returns:
                # "p(X) or p(X | C) if number of classes > 1"
                # We treat the model output as log P(X|Z). We need to marginalize Z.

                # Assume uniform prior for training stability or learnable prior
                # Simple-einet MixingLayer usually handles this, but here we access raw class outputs
                log_prob = torch.logsumexp(ll_output, dim=1)
            else:
                log_prob = ll_output

            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

        self.model.eval()
        logging.info(
            f"Client {self.client_id} finished training. Final Loss: {loss.item():.4f}"
        )
        return self


class ServerSPN(FederatedSPNBase):
    """
    Server Node: The Federated Circuit (FC) Orchestrator.
    Implements the 'Arithmetic' to combine client distributions.
    """

    def __init__(
        self,
        global_num_features: int,
        scenario: str = "horizontal",
        num_clusters: int = 1,  # K (Latent clusters)
        device="cpu",
        threshold: float = 0.01,
    ):
        super().__init__(device)
        self.global_num_features = global_num_features
        self.scenario = scenario
        self.num_clusters = num_clusters  # Should match Client config
        self.threshold = threshold

        self.clients: List[ClientSPN] = []
        self.client_weights: List[float] = []  # For Horizontal mixing
        self.feature_map: Dict[
            int, List[int]
        ] = {}  # Maps client_id -> [feature_indices]

    def register_client(self, client: ClientSPN, feature_indices: List[int]):
        """
        Registers a trained client.
        feature_indices: The list of global feature indices this client owns.
        """
        self.clients.append(client)
        self.feature_map[id(client)] = feature_indices
        self.client_weights.append(client.data_count)

    def ci_test(
        self,
        x_idx: Union[int, List[int]],
        y_idx: Union[int, List[int]],
        z_idx: List[int],
        data_matrix: np.ndarray,
    ) -> float:
        """
        The Oracle Function used by CDNOD.
        Computes CMI(X; Y | Z) using the Federated Circuit.

        Args:
            x_idx, y_idx: Target variables
            z_idx: Conditioning variables
            data_matrix: Proxy data (e.g. from the test set) to evaluate expectations.
        """
        if isinstance(x_idx, (int, np.integer)):
            x_idx = [x_idx]
        if isinstance(y_idx, (int, np.integer)):
            y_idx = [y_idx]
        if z_idx is None:
            z_idx = []
        z_idx = list(z_idx)

        # Convert data to tensor once
        if not isinstance(data_matrix, torch.Tensor):
            data_tensor = torch.tensor(data_matrix, dtype=torch.float32).to(self.device)
        else:
            data_tensor = data_matrix

        # CMI(X;Y|Z) ≈ E[ log P(X|Y,Z) - log P(X|Z) ]
        #            = E[ log P(X,Y,Z) - log P(Y,Z) - (log P(X,Z) - log P(Z)) ]
        #            = E[ log P(XYZ) - log P(XZ) - log P(YZ) + log P(Z) ]

        ll_xyz = self._federated_inference(data_tensor, x_idx + y_idx + z_idx)
        ll_xz = self._federated_inference(data_tensor, x_idx + z_idx)
        ll_yz = self._federated_inference(data_tensor, y_idx + z_idx)

        if z_idx:
            ll_z = self._federated_inference(data_tensor, z_idx)
        else:
            ll_z = 0.0

        # Estimate CMI from the sample average log-likelihoods
        cmi_est = ll_xyz - ll_xz - ll_yz + ll_z

        # Ensure non-negative (theoretical constraint, though estimation noise exists)
        cmi_est = max(0.0, cmi_est)

        # Thresholding
        is_independent = cmi_est < self.threshold

        # Return p-value style (1.0 = Indep, 0.0 = Dep)
        return 1.0 if is_independent else 0.0

    def _federated_inference(self, data: torch.Tensor, scope: List[int]) -> float:
        """
        Computes mean log-likelihood E[log P(scope)] using the Federated Circuit structure.
        Handles Horizontal (Sum) and Vertical (Mixture of Products).
        """
        if self.scenario == "horizontal":
            return self._inference_horizontal(data, scope)
        elif self.scenario in ["vertical", "hybrid"]:
            # Both Vertical and Hybrid have feature subsets -> Use Product/Slicing logic
            return self._inference_vertical(data, scope)
        return 0.0

    def _get_client_log_prob(
        self, client: ClientSPN, global_data: torch.Tensor, global_scope: List[int]
    ):
        """
        Helper: Slices global data to match client's local features.
        """
        client_feats = self.feature_map[id(client)]
        batch_size = global_data.shape[0]

        # 1. Create a local tensor of the correct shape [N, D_local] with NaNs
        # This handles the "200 vs 4" mismatch error.
        local_data = torch.full(
            (batch_size, len(client_feats)), float("nan"), device=self.device
        )

        # 2. Identify intersection of request scope and client features
        # Map Global Index -> Local Index
        local_scope_indices = []
        for local_idx, global_idx in enumerate(client_feats):
            if global_idx in global_scope:
                # Fill observed data
                local_data[:, local_idx] = global_data[:, global_idx]
                local_scope_indices.append(local_idx)

        # 3. If client covers NONE of the requested variables, return 0.0 (Log Likelihood of 1)
        # However, for product aggregation, we might need valid shape.
        # simple-einet handles all-nan input as marginalizing everything -> 0.0 log prob.

        with torch.no_grad():
            return client.model(local_data)

    def _inference_horizontal(self, data: torch.Tensor, scope: List[int]) -> float:
        """
        Model: P(V) = Sum_k w_k P_k(V) (Mixture)
        """
        # 1. Weights
        total_count = sum(self.client_weights)
        log_weights = torch.tensor(
            [np.log(w / total_count) for w in self.client_weights], device=self.device
        )  # [K_clients]

        # 2. Query Clients
        # Each client returns [N, 1] log-probs
        client_lls = []
        for client in self.clients:
            # Horizontal: All clients have all features. Scope matches.
            ll = self._get_client_log_prob(client, data, scope)
            client_lls.append(ll)  # List of [N, 1]

        # Stack: [N, K_clients]
        stacked_lls = torch.cat(client_lls, dim=1)

        # 3. Mixture LogSumExp
        # log( Sum w_k exp(ll_k) )
        weighted_lls = stacked_lls + log_weights.unsqueeze(0)
        final_ll = torch.logsumexp(weighted_lls, dim=1)

        return final_ll.mean().item()

    def _inference_vertical(self, data: torch.Tensor, scope: List[int]) -> float:
        """
        Model: P(V) = Sum_c Prior(c) * Prod_k P_k(V_k | c) (Mixture of Products)
        Paper Assumption 2.
        """
        # Prior over clusters (Latent L). Assuming uniform for now (1/K)
        # or we could learn this.
        log_prior = -np.log(self.num_clusters)

        # We need to accumulate log-probs for each cluster c
        # Result shape goal: [N, Num_Clusters]

        # Initialize with Prior: [N, Num_Clusters]
        # We start with 0.0 (log(1)) because we are doing Product (Sum of Logs)
        batch_size = data.shape[0]
        cluster_accumulators = torch.zeros(
            (batch_size, self.num_clusters), device=self.device
        )

        for client in self.clients:
            # Query Client (Returns [N, K])
            client_cluster_lls = self._get_client_log_prob(client, data, scope)

            # Product Rule in Log Space = Sum
            cluster_accumulators += client_cluster_lls

            # Marginalize Latent Cluster
        final_vals = torch.logsumexp(cluster_accumulators + log_prior, dim=1)
        return final_vals.mean().item()
