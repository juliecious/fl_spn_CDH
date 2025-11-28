import logging

import numpy as np
from causallearn.utils.data_utils import (
    get_cpdag_from_cdnod,
    get_dag_from_pdag,
    count_skeleton_accuracy,
    count_dag_accuracy,
)

from causallearn.search.ConstraintBased.CDNOD import cdnod
from test_data import X_raw as X, K, sample_num as n, d, true_DAG_bin

c_indx = np.asarray(list(range(K)))
c_indx = np.repeat(c_indx, n)
c_indx = np.reshape(c_indx, (n * K, 1))

cg = cdnod(
    X,
    c_indx,
    K,
    alpha=0.05,
    indep_test="spn",
    stable=True,
    uc_rule=0,
    uc_priority=-1,
)

est_graph = np.zeros((d, d))
est_graph = cg.G.graph[0:d, 0:d]

est_cpdag = get_cpdag_from_cdnod(
    est_graph
)  # est_graph[i,j]=-1 & est_graph[j,i]=1  ->  est_graph_cpdag[i,j]=1
est_dag_from_pdag = get_dag_from_pdag(
    est_cpdag
)  # return a DAG from a PDAG in causaldag.

# Undirected skeleton: F1, recall, precision, SHD
ret_skeleton = count_skeleton_accuracy(true_DAG_bin, est_cpdag)
logging.info(f"Undirected skeleton:\n{ret_skeleton}")

# Directed graph: F1, recall, precision, SHD
ret_diretion = count_dag_accuracy(true_DAG_bin, est_dag_from_pdag)
logging.info(f"Directed graph:\n{ret_diretion}")

# if __name__ == "__main__":
#     logging.info(cg)
