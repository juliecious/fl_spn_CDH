#!/usr/bin/env python3
"""
Test: Verify initial skeleton loading by CLEARING the graph first.
"""

import numpy as np
from causallearn.graph.GraphClass import CausalGraph

print("=" * 80)
print("TEST: Loading initial skeleton (clearing graph first)")
print("=" * 80)

n_vars = 8
n_edges_target = 22

# Create initial skeleton
initial_skeleton = np.zeros((n_vars, n_vars), dtype=int)
edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
][:n_edges_target]

for i, j in edges:
    initial_skeleton[i, j] = 1
    initial_skeleton[j, i] = 1

expected_edges = initial_skeleton.sum() // 2
print(f"Initial skeleton matrix: {expected_edges} edges")

# OLD WAY (doesn't work):
print("\n--- OLD WAY (without clearing) ---")
cg_old = CausalGraph(no_of_var=n_vars)
print(f"Fresh CausalGraph: {cg_old.G.get_num_edges()} edges (fully connected)")

for node_i in range(n_vars):
    for node_j in range(node_i + 1, n_vars):
        if initial_skeleton[node_i, node_j] == 1:
            cg_old.G.graph[node_i, node_j] = -1
            cg_old.G.graph[node_j, node_i] = -1

print(f"After loading skeleton (OLD): {cg_old.G.get_num_edges()} edges")
print(f"❌ WRONG: Still fully connected (28 edges), not {expected_edges} edges")

# NEW WAY (with clearing):
print("\n--- NEW WAY (with clearing first) ---")
cg_new = CausalGraph(no_of_var=n_vars)
print(f"Fresh CausalGraph: {cg_new.G.get_num_edges()} edges (fully connected)")

# CLEAR the graph first
cg_new.G.graph[:] = 0
print(f"After clearing: {cg_new.G.get_num_edges()} edges")

# Now load skeleton
for node_i in range(n_vars):
    for node_j in range(node_i + 1, n_vars):
        if initial_skeleton[node_i, node_j] == 1:
            cg_new.G.graph[node_i, node_j] = -1
            cg_new.G.graph[node_j, node_i] = -1

actual_edges = cg_new.G.get_num_edges()
print(f"After loading skeleton (NEW): {actual_edges} edges")

if actual_edges == expected_edges:
    print(f"✅ CORRECT: {actual_edges} edges matches {expected_edges} expected")
else:
    print(f"❌ WRONG: {actual_edges} edges != {expected_edges} expected")

print("\n" + "=" * 80)
if actual_edges == expected_edges:
    print("TEST PASSED ✓")
else:
    print("TEST FAILED ✗")
print("=" * 80)
