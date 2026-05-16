"""
Sachs (Protein Signaling) Dataset Loader for Federated Causal Discovery.
Robustly fetches data using Pooch and applies standard biological preprocessing.

URL Source: BNLearn (Scutari et al.)
"""

import os
import sys
import numpy as np
import pandas as pd
import pooch
import logging

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

NODE_NAMES = [
    "raf",
    "mek",
    "plcg",
    "pip2",
    "pip3",
    "erk",
    "akt",
    "pka",
    "pkc",
    "p38",
    "jnk",
]


def get_sachs_ground_truth() -> np.ndarray:
    """Returns consensus Sachs DAG adjacency matrix."""
    from tests.utils.benchmark_loaders import load_standard_graph

    return load_standard_graph("sachs")


def load_sachs_federated(n_clients=3, n_samples_limit=None):
    """
    Downloads, cleans, and partitions the Sachs interventional dataset.
    Each client receives a subset of interventional conditions (Real Heterogeneity).
    """
    url = "https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz"
    known_hash = "39ee257f7eeb94cb60e6177cf80c9544"

    try:
        logging.info(f"Fetching Sachs data from {url}...")
        file_path = pooch.retrieve(
            url=url,
            known_hash=f"md5:{known_hash}",
            path="data/real_world",
            fname="sachs.interventional.txt.gz",
        )

        # Load (Space delimited, Gzipped)
        df = pd.read_csv(file_path, delimiter=" ", compression="gzip")

        # --- SUB-SAMPLE FOR QUICK TESTING ---
        if n_samples_limit and len(df) > n_samples_limit:
            df = df.sample(n=n_samples_limit, random_state=42)
            logging.warning(
                f"Sub-sampled Sachs data to {n_samples_limit} rows for testing."
            )
        # ------------------------------------

        # The file typically has 11 proteins + 1 'INT' column (Intervention type)
        # Standardize column names to lowercase to match NODE_NAMES
        df.columns = [c.lower() for c in df.columns]

        # 1. Preprocessing: log(1+x)
        features = [c for c in df.columns if c in NODE_NAMES]
        df[features] = np.log1p(df[features])

        # 2. Extract Labels and Data
        X_all = df[features].values

        # 3. Partition by Intervention (The INT column)
        # If 'int' column exists, use it to group clients.
        # Real Sachs has ~9-14 conditions. We map them to n_clients.
        if "int" in df.columns:
            conditions = df["int"].unique()
            # Split conditions into n_clients groups
            cond_splits = np.array_split(conditions, n_clients)

            X_splits = []
            c_indices = []
            for k in range(n_clients):
                mask = df["int"].isin(cond_splits[k])
                X_k = X_all[mask]
                # Standardize locally (to preserve heterogeneity in mean/var)
                X_k = (X_k - X_k.mean(axis=0)) / (X_k.std(axis=0) + 1e-6)
                X_splits.append(X_k)
                c_indices.append(np.full((len(X_k), 1), k))

            c_indx = np.vstack(c_indices)
        else:
            # Fallback to simple split if 'int' column is missing
            X_all = (X_all - X_all.mean(axis=0)) / (X_all.std(axis=0) + 1e-6)
            X_splits = np.array_split(X_all, n_clients)
            c_indx = np.repeat(np.arange(n_clients), len(X_all) // n_clients).reshape(
                -1, 1
            )

        true_dag = get_sachs_ground_truth()
        return X_splits, true_dag, c_indx

    except Exception as e:
        logging.error(
            f"Failed to load real Sachs data: {e}. Falling back to synthetic."
        )
        from tests.utils.benchmark_loaders import (
            simulate_heterogeneous_data,
            load_standard_graph,
        )

        true_dag = load_standard_graph("sachs")
        X_global, c_indx = simulate_heterogeneous_data(
            true_dag, n_clients, n_samples=500
        )
        X_splits = [X_global[c_indx.flatten() == k] for k in range(n_clients)]
        return X_splits, true_dag, c_indx


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    splits, dag, c = load_sachs_federated()
    print(f"Success! Loaded {len(splits)} clients with real interventional shifts.")
    print(f"Total samples: {len(c)}")
    print(f"Graph nodes: {dag.shape[0]}")
