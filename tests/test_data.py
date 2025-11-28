import logging

import numpy as np
from causallearn.utils.cit import CIT
import torch
from causallearn.utils.data_utils import simulate_dag, my_simulate_linear_gaussian
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Generate a centralized test data
d, s0, K = 8, 2, 1
sample_num = 5_000
graph_type, sem_type = "ER", "gauss"
true_DAG_bin = simulate_dag(d, s0, graph_type)  # 產生隨機因果結構

X_raw, _ = my_simulate_linear_gaussian(
    true_DAG_bin, K=1, n=sample_num, sem_type=sem_type
)
# logging.info(X_raw.shape)

scaler = StandardScaler()
X_proc = scaler.fit_transform(X_raw)
logging.info(f"Data Shape: {X_proc.shape} (Standardized)")

# logging.info(X)
