"""
Federated SPN implementations.

This package contains implementations for different federated learning scenarios:
- Horizontal: Same features, different samples
- Vertical: Different features, same samples
- Hybrid: Mixed horizontal and vertical partitioning
- Cluster-conditional: Vertical with cluster-based conditioning

Gap Fixes (NEW):
- Structure Sync: Ensures structural alignment across clients (Gap 1)
- Latent Routing: Captures cross-silo dependencies (Gap 2)
"""

from .horizontal import GlobalFedSPN
from .vertical import FederatedProduct, ProductOverGroups, GroupMixture
from .hybrid import ProductOverGroupsWithOverlap, GlobalSumOfProducts
from .cluster_conditional import FederatedProductWithClusters

# Gap Fixes (NEW)
from .structure_sync import (
    FederatedPCStructureManager,
    broadcast_structure_to_clients,
)
from .latent_routing import (
    LatentFederatedProductNode,
    replace_product_with_latent_routing,
)

__all__ = [
    "GlobalFedSPN",
    "FederatedProduct",
    "ProductOverGroups",
    "GroupMixture",
    "ProductOverGroupsWithOverlap",
    "GlobalSumOfProducts",
    "FederatedProductWithClusters",
    # Gap Fixes
    "FederatedPCStructureManager",
    "broadcast_structure_to_clients",
    "LatentFederatedProductNode",
    "replace_product_with_latent_routing",
]
