"""
reproduce_feature_dictionary.py
==============================================================================
Data dictionary for the 63 topology features fed to the RandomForest models
(Phases 20-22 and the reproduction's "full (all topology)" set). The feature
list is DERIVED from build_feature_table() with the exact same META/DROP rules
the models use, then matched against hand-written descriptions -- the script
asserts the two agree, so the dictionary can never silently drift from the model
input.

Nodes are the three genes x1, x2, x3; the adjacency matrix has entries in
{-1 (repress), 0 (none), +1 (activate)}, row = source, col = target.

Output: results/reproduce/topology_feature_dictionary.csv  (feature, group, description)
Run:    python reproduce_feature_dictionary.py
==============================================================================
"""

import os
import pandas as pd

from repro_common import build_feature_table
from motif_lib import DEPRECATED_FEATURE_COLS

os.makedirs("results/reproduce", exist_ok=True)

# --- the exact 63-feature model input (same derivation as the RF scripts) ----
feat = build_feature_table()
META = ["net_id", "label", "multimodal_count", "multimodal_rate",
        "label_t05", "label_t10", "label_t25"]
DROP = set(DEPRECATED_FEATURE_COLS) | {"edge_density", "n_zero"} | \
       {c for c in feat.columns if c.startswith("has_")} | \
       {"fb_pos_feedback", "fb_neg_feedback", "fb_total_feedback"}
FULL = [c for c in feat.columns if c not in META and c not in DROP]

# --- descriptions (grouped) ---------------------------------------------------
GRP, DESC = {}, {}
def add(f, g, d):
    GRP[f], DESC[f] = g, d

for i in (1, 2, 3):
    add(f"self_reg_{i}", "Structure",
        f"Self-regulation sign of gene x{i} (diagonal; -1 repress / 0 none / +1 activate)")
for i, j in [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]:
    add(f"edge_{i}{j}", "Structure", f"Directed edge sign x{i} -> x{j} (-1 / 0 / +1)")
add("n_activating", "Structure", "Count of activating entries (+1) in the matrix (incl. self-loops)")
add("n_repressing", "Structure", "Count of repressing entries (-1) in the matrix (incl. self-loops)")
add("n_edges", "Structure", "Total nonzero entries = number of edges incl. self-loops (network density)")
add("n_self_activating", "Structure", "Number of activating self-loops (diagonal = +1)")
add("n_self_repressing", "Structure", "Number of repressing self-loops (diagonal = -1)")
add("n_self_none", "Structure", "Number of genes without self-regulation (diagonal = 0)")
add("n_off_activating", "Structure", "Number of activating off-diagonal edges")
add("n_off_repressing", "Structure", "Number of repressing off-diagonal edges")
add("n_off_zero", "Structure", "Number of absent off-diagonal edges")

for i in (1, 2, 3):
    add(f"in_deg_pos_{i}", "Degree", f"Activating incoming edges to x{i} (self-regulation excluded)")
    add(f"in_deg_neg_{i}", "Degree", f"Repressing incoming edges to x{i}")
    add(f"out_deg_pos_{i}", "Degree", f"Activating outgoing edges from x{i}")
    add(f"out_deg_neg_{i}", "Degree", f"Repressing outgoing edges from x{i}")
    add(f"in_deg_total_{i}", "Degree", f"Total incoming edges to x{i} (nonzero, self excluded)")
    add(f"out_deg_total_{i}", "Degree", f"Total outgoing edges from x{i} (nonzero, self excluded)")
    add(f"net_in_deg_{i}", "Degree", f"Net activating input to x{i} (in_deg_pos - in_deg_neg)")
add("max_in_degree", "Degree", "Maximum in-degree across the three genes")
add("max_out_degree", "Degree", "Maximum out-degree across the three genes")
add("mean_in_degree", "Degree", "Mean in-degree across the three genes")
add("mean_out_degree", "Degree", "Mean out-degree across the three genes")

add("fb_pos_1node", "Feedback", "Positive 1-node feedback loops (activating self-loops)")
add("fb_neg_1node", "Feedback", "Negative 1-node feedback loops (repressing self-loops)")
add("fb_pos_2node", "Feedback", "Positive 2-node feedback loops (mutual pair, edge-sign product +1)")
add("fb_neg_2node", "Feedback", "Negative 2-node feedback loops (mutual pair, edge-sign product -1)")
add("fb_pos_3node", "Feedback", "Positive 3-node feedback loops (3-cycle, edge-sign product +1)")
add("fb_neg_3node", "Feedback", "Negative 3-node feedback loops (3-cycle, edge-sign product -1)")

add("n_scc", "Connectivity", "Number of strongly connected components (directed graph)")
add("largest_scc_size", "Connectivity", "Size of the largest SCC (1-3; 3 = fully strongly connected)")
add("is_strongly_connected", "Connectivity", "1 if the whole graph is a single SCC, else 0")
add("n_wcc", "Connectivity", "Number of weakly connected components (edge direction ignored)")
add("is_weakly_connected", "Connectivity", "1 if the graph is weakly connected, else 0")
add("reachability", "Connectivity", "Fraction of ordered gene pairs (i!=j) with a directed path i -> j")

add("trace", "Spectral", "Trace of the adjacency matrix (= sum of self-regulations)")
add("det", "Spectral", "Determinant of the adjacency matrix")
add("spectral_radius", "Spectral", "Largest eigenvalue magnitude, max|lambda|")
add("max_eigenvalue_real", "Spectral", "Largest real part among the eigenvalues")
add("min_eigenvalue_real", "Spectral", "Smallest real part among the eigenvalues")
add("sum_eigenvalue_real", "Spectral", "Sum of eigenvalue real parts (equals the trace)")
add("eigenvalue_spread", "Spectral", "max_eigenvalue_real - min_eigenvalue_real")
add("frobenius_norm", "Spectral", "Frobenius norm of the matrix (overall interaction strength)")

# --- self-validation: descriptions must match the model's feature set exactly -
missing = [f for f in FULL if f not in DESC]
extra = [f for f in DESC if f not in FULL]
assert not missing and not extra, f"dictionary out of sync -> missing={missing}, extra={extra}"

tbl = pd.DataFrame({"feature": FULL,
                    "group": [GRP[f] for f in FULL],
                    "description": [DESC[f] for f in FULL]})
tbl.to_csv("results/reproduce/topology_feature_dictionary.csv", index=False)
print(f"{len(tbl)} topology features documented "
      f"({tbl.group.value_counts().reindex(['Structure','Degree','Feedback','Connectivity','Spectral']).to_dict()})")
print("Saved -> results/reproduce/topology_feature_dictionary.csv")
