#!/usr/bin/env python3
"""
Quick test to verify initial skeleton is being loaded correctly.
"""

import numpy as np
from causallearn.graph.GraphClass import CausalGraph

# Test 1: Create a CausalGraph with initial skeleton
print("=" * 80)
print("TEST: Creating CausalGraph with initial skeleton")
print("=" * 80)

n_vars = 8
n_edges = 22

# Create initial skeleton
initial_skeleton = np.zeros((n_vars, n_vars), dtype=int)
# Add some edges
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
][:n_edges]

for i, j in edges:
    initial_skeleton[i, j] = 1
    initial_skeleton[j, i] = 1

actual_edges = initial_skeleton.sum() // 2
print(f"Created initial skeleton matrix with {actual_edges} edges")

# Load into CausalGraph
cg = CausalGraph(no_of_var=n_vars)
for node_i in range(n_vars):
    for node_j in range(node_i + 1, n_vars):
        if initial_skeleton[node_i, node_j] == 1:
            cg.G.graph[node_i, node_j] = -1
            cg.G.graph[node_j, node_i] = -1

# Check edges
n_edges_loaded = cg.G.get_num_edges()
print(f"CausalGraph.G.get_num_edges() = {n_edges_loaded}")
print(f"Expected: {actual_edges * 2} (each edge appears twice in adjacency matrix)")

# Check if condition would trigger in skeleton_discovery
cg_list = [cg]
if cg_list and len(cg_list) > 0 and cg_list[0].G.get_num_edges() > 0:
    print(f"✓ Condition TRUE: Would use initial skeleton")
else:
    print(f"✗ Condition FALSE: Would create fresh graph")

print("\n" + "=" * 80)
print("TEST RESULT")
print("=" * 80)
if n_edges_loaded > 0:
    print(f"✓ PASS: Initial skeleton loaded successfully")
else:
    print(f"✗ FAIL: No edges loaded")
