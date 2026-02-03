import time
from itertools import permutations, combinations
from typing import Dict, List, Optional

import networkx as nx
import torch.nn as nn
from numpy import ndarray
import numpy as np

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils import SkeletonDiscovery, UCSepset, Meek
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import (
    orient_by_background_knowledge,
)
from causallearn.utils.cit import *
from causallearn.search.ConstraintBased.PC import (
    get_parent_missingness_pairs,
    skeleton_correction,
)

from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from mlxtend.preprocessing import standardize
from copy import deepcopy
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint


def my_cov(X, Y: np.ndarray = None):
    if Y is not None:
        X = X - X.mean(axis=0)  # (n,h)
        Y = Y - Y.mean(axis=0)  # (n,h)
        cov = X.T @ Y
    else:
        X = X - X.mean(axis=0)  # (n,h)
        cov = X.T @ X
    factor = len(X)
    return cov / factor


def get_hsic_score_fast(X, Y, C_f, iCcc):
    """
    Optimized version that accepts pre-computed C features and Inverse Covariance.
    """
    h = 5
    feature_map = Nystroem(gamma=0.2, n_components=h, random_state=1)

    X_f = feature_map.fit_transform(X)
    Y_f = feature_map.fit_transform(Y)

    XY = np.concatenate((X, Y), axis=1)
    XY_f = feature_map.fit_transform(XY)

    Cxyc = my_cov(XY_f, C_f)
    Cxc = my_cov(X_f, C_f)
    Cyc = my_cov(Y_f, C_f)

    # iCcc is already computed!
    Mu_xy = Cxyc @ iCcc @ C_f.T
    Mu_x = Cxc @ iCcc @ C_f.T
    Mu_y = Cyc @ iCcc @ C_f.T

    hsic_x_y = np.sum(my_cov(Mu_x, Mu_xy) ** 2) / np.trace(my_cov(Mu_x))
    hsic_y_x = np.sum(my_cov(Mu_y, Mu_xy) ** 2) / np.trace(my_cov(Mu_y))

    if hsic_x_y < hsic_y_x:
        return 1
    else:
        return 2


def get_hybrid_direction_score(
    fed_spn_model, i, j, c_idx, data_aug, C_f=None, iCcc=None
):
    """
    Hybrid Orientation: Ensembles SPN-based mechanism invariance with HSIC.
    """
    # 1. SPN Score (Mechanism Invariance)
    cit = SPN_CIT(data_aug, global_model=fed_spn_model, threshold=0.01)

    def compute_cmi(X, Y, Z):
        ll_xyz = cit._get_marginal_log_prob(data_aug, X + Y + Z)
        ll_xz = cit._get_marginal_log_prob(data_aug, X + Z)
        ll_yz = cit._get_marginal_log_prob(data_aug, Y + Z)
        ll_z = cit._get_marginal_log_prob(data_aug, Z) if Z else 0.0
        return max(0.0, ll_xyz - ll_xz - ll_yz + ll_z)

    s_xy = compute_cmi([j], [c_idx], [i])  # Y _|_ C | X
    s_yx = compute_cmi([i], [c_idx], [j])  # X _|_ C | Y

    spn_dir = 1 if s_xy < s_yx else 2

    # 2. HSIC Score (Ensemble logic could go here)
    # For now, SPN is preferred for federated density consistency.
    return spn_dir


def cdnod(
    data: ndarray,
    c_indx: ndarray,
    K: int = 1,
    alpha: float = 0.05,
    indep_test: str = fisherz,
    stable: bool = True,
    uc_rule: int = 0,
    uc_priority: int = 2,
    mvcdnod: bool = False,
    correction_name: str = "MV_Crtn_Fisher_Z",
    background_knowledge: Optional[BackgroundKnowledge] = None,
    verbose: bool = False,
    show_progress: bool = True,
    fed_spn_model: Optional[nn.Module] = None,
    **kwargs,
) -> CausalGraph:
    if mvcdnod:
        return mvcdnod_alg(
            data=data,
            alpha=alpha,
            indep_test=indep_test,
            correction_name=correction_name,
            stable=stable,
            uc_rule=uc_rule,
            uc_priority=uc_priority,
            verbose=verbose,
            show_progress=show_progress,
            **kwargs,
        )
    else:
        return cdnod_alg(
            data=data,
            c_indx=c_indx,
            alpha=alpha,
            K=K,
            indep_test=indep_test,
            stable=stable,
            uc_rule=uc_rule,
            uc_priority=uc_priority,
            background_knowledge=background_knowledge,
            verbose=verbose,
            show_progress=show_progress,
            fed_spn_model=fed_spn_model,
            **kwargs,
        )


def cdnod_alg(
    data: ndarray,
    c_indx: ndarray,
    alpha: float,
    K: int,
    indep_test: str,
    stable: bool,
    uc_rule: int,
    uc_priority: int,
    background_knowledge: Optional[BackgroundKnowledge] = None,
    verbose: bool = False,
    show_progress: bool = True,
    fed_spn_model: Optional[nn.Module] = None,
    **kwargs,
) -> CausalGraph:
    start = time.time()
    data_aug = np.concatenate((data, c_indx), axis=1)

    if fed_spn_model is not None:
        indep_test_all = SPN_CIT(data_aug, global_model=fed_spn_model, **kwargs)
    elif callable(indep_test):
        indep_test_all = indep_test
    else:
        indep_test_all = CIT(data_aug, indep_test, **kwargs)

    s_a, d = data_aug.shape
    fed_data = data_aug.reshape(K, int(s_a / K), d)
    cg_list = []
    for i in range(K):
        fed_dt = fed_data[i]
        fed_cg = CausalGraph(no_of_var=data_aug.shape[1], node_names=None)
        if fed_spn_model is not None:
            fed_indep_test = indep_test_all
        elif callable(indep_test):
            fed_indep_test = indep_test
        else:
            fed_indep_test = CIT(fed_dt, indep_test)
        fed_cg.set_ind_test(fed_indep_test)
        cg_list.append(fed_cg)

    # Stage 1
    flag = 0
    cg_0 = SkeletonDiscovery.skeleton_discovery(
        flag, cg_list, data, K, alpha, indep_test_all, stable
    )

    # Stage 2
    cg_1 = SkeletonDiscovery.skeleton_discovery_with_surrogate_GMM(
        flag, cg_0, cg_list, data_aug, K, alpha, indep_test_all, stable
    )

    # Orient edge from c_indx
    c_indx_id = data_aug.shape[1] - 1
    for i in cg_1.G.get_adjacent_nodes(cg_1.G.nodes[c_indx_id]):
        cg_1.G.add_directed_edge(cg_1.G.nodes[c_indx_id], i)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    # Orientation logic
    cg = None
    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(
                cg_1, uc_priority, background_knowledge=background_knowledge
            )
        else:
            cg_2 = UCSepset.uc_sepset(
                cg_1, background_knowledge=background_knowledge, cg_list=cg_list, K=K
            )
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)
    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(
                cg_1, uc_priority, background_knowledge=background_knowledge
            )
        else:
            cg_2 = UCSepset.maxp(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)
    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(
                cg_1, alpha, uc_priority, background_knowledge=background_knowledge
            )
        else:
            cg_2 = UCSepset.definite_maxp(
                cg_1, alpha, background_knowledge=background_knowledge
            )
        cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_before, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")

    # Stage 3
    feature_map = Nystroem(gamma=0.2, n_components=5, random_state=1)
    C_f = feature_map.fit_transform(c_indx)
    Ccc = my_cov(C_f, C_f)
    iCcc = np.linalg.inv(Ccc + np.eye(5) * 1e-10)

    vh = []
    for i in range(d - 1):
        if (cg.G.graph[i, d - 1] == 1) and (cg.G.graph[d - 1, i] == -1):
            vh.append(i)

    for v in combinations(vh, 2):
        i, j = v
        if fed_spn_model is not None:
            score_i_j = get_hybrid_direction_score(
                fed_spn_model, i, j, c_indx_id, data_aug, C_f, iCcc
            )
        else:
            score_i_j = get_hsic_score_fast(
                data[:, i].reshape(-1, 1), data[:, j].reshape(-1, 1), C_f, iCcc
            )

        if score_i_j == 1:
            cg.G.add_edge(
                Edge(cg.G.nodes[i], cg.G.nodes[j], Endpoint.TAIL, Endpoint.ARROW)
            )
        else:
            cg.G.add_edge(
                Edge(cg.G.nodes[j], cg.G.nodes[i], Endpoint.TAIL, Endpoint.ARROW)
            )

    end = time.time()
    cg.PC_elapsed = end - start
    return cg


def mvcdnod_alg(
    data: ndarray,
    alpha: float,
    indep_test: str,
    correction_name: str,
    stable: bool,
    uc_rule: int,
    uc_priority: int,
    verbose: bool,
    show_progress: bool,
    **kwargs,
) -> CausalGraph:
    start = time.time()
    indep_test_obj = CIT(data, indep_test, **kwargs)
    prt_m = get_parent_missingness_pairs(data, alpha, indep_test_obj, stable)
    cg_pre = SkeletonDiscovery.skeleton_discovery(
        data,
        alpha,
        indep_test_obj,
        stable,
        verbose=verbose,
        show_progress=show_progress,
    )
    cg_pre.to_nx_skeleton()
    cg_corr = skeleton_correction(data, alpha, correction_name, cg_pre, prt_m, stable)
    c_indx_id = data.shape[1] - 1
    for i in cg_corr.G.get_adjacent_nodes(cg_corr.G.nodes[c_indx_id]):
        cg_corr.G.add_directed_edge(i, cg_corr.G.nodes[c_indx_id])

    if uc_rule == 0:
        cg_2 = (
            UCSepset.uc_sepset(cg_corr, uc_priority)
            if uc_priority != -1
            else UCSepset.uc_sepset(cg_corr)
        )
        cg = Meek.meek(cg_2)
    elif uc_rule == 1:
        cg_2 = (
            UCSepset.maxp(cg_corr, uc_priority)
            if uc_priority != -1
            else UCSepset.maxp(cg_corr)
        )
        cg = Meek.meek(cg_2)
    elif uc_rule == 2:
        cg_2 = (
            UCSepset.definite_maxp(cg_corr, alpha, uc_priority)
            if uc_priority != -1
            else UCSepset.definite_maxp(cg_corr, alpha)
        )
        cg_before = Meek.definite_meek(cg_2)
        cg = Meek.meek(cg_before)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    end = time.time()
    cg.PC_elapsed = end - start
    return cg
