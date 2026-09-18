"""
repro_common.py — shared helpers for standalone reproduction scripts.
==============================================================================
Builds the per-network analysis table from the TWO raw inputs only
    1. network_data.json                       (networks + adjacency matrices)
    2. multimodal_results_combined_sorted.csv   (or .xlsx) (successful runs)
and provides the statistics reused across reproduction scripts:
    - crude + density-controlled (Mantel-Haenszel / CMH) odds ratios   [identity]
    - binomial-GLM per-loop log-odds (quasi-binomial CIs)              [robustness]

No dependence on any derived file (e.g. results/features.csv). Only code
dependency: motif_lib.py. Used by:
    reproduce_mh_table_standalone.py   (PFL/NFL/coherent-FFL/incoherent-FFL table)
    reproduce_feedback_roles.py        (6 elementary fb categories: identity + %)
==============================================================================
"""

import os
import re
import json
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import StratifiedTable

from motif_lib import (feedback_counts, feedforward_counts, motif_contingency,
                       FEEDBACK_CATEGORIES)

# --- constants reused by reproduction scripts -------------------------------
ELEMENTARY = [f"fb_{c}" for c in FEEDBACK_CATEGORIES]   # fb_pos_1node ... fb_neg_3node
N_TRIALS_DEFAULT = 1000
# Canonical BASELINE results file (noise level 1.0), one row per multimodal (net_id, param_id).
# Updated 2026-08-17 simulation. Schema: single system-level `stability_time`, per-gene
# `cv_before_x{1,2,3}` / `cv_after_x{1,2,3}`, plus `accepted_nodes` / `discarded_node`
# (active-gene bookkeeping). Topology (net_id -> adjacency) is unchanged and matches both
# network_data.json and unique_networks_table.jld2 (verified byte-identical).
# Canonical results set: sigma 1.0, 10% minimum state fraction AND reproducible from
# opposite initial conditions. Superseded summary_result_noise1_0817.csv (the 5%-floor
# 2026-08-17 run) as the default on 2026-09-17; that file is no longer read by any
# pipeline step, only by the upstream filters that produced these six inputs.
BASELINE_RESULTS_FILE = "summary_result_noise1_frac10_opicFiltered.csv"

# --- results-file switch (env HR_RESULTS) -----------------------------------
# Point HR_RESULTS at any of the six filtered results files to run a reproduction
# against it; outputs are tagged with the variant's name so the runs never clobber
# one another:
#     HR_RESULTS=summary_result_noise5_frac10_opicFiltered.csv python reproduce_hr_composition.py
#         -> results/reproduce/hr_composition_noise5_frac10_opic.{csv,png,svg}
# The tag drops the shared "summary_result_" prefix and, for the sigma-1 files only,
# the redundant "noise1_", so the six inputs map to:
#     noise1_frac10       -> "_frac10"           noise1_frac10_opic  -> "_frac10_opic"
#     noise02_frac10      -> "_noise02_frac10"   noise02_frac10_opic -> "_noise02_frac10_opic"
#     noise5_frac10       -> "_noise5_frac10"    noise5_frac10_opic  -> "_noise5_frac10_opic"
# The tag is UNCONDITIONAL -- it is derived from the file name alone, including for the
# default. Previously the default wrote untagged names, which meant the canonical
# figures carried no record of which results set produced them.
RESULTS_NAME = os.environ.get("HR_RESULTS", BASELINE_RESULTS_FILE)
_STEM = os.path.splitext(os.path.basename(RESULTS_NAME))[0]
# Qualifiers that make an INPUT file name self-explanatory but would only lengthen the
# OUTPUT tag are normalised away, so the figure names are unchanged by a file rename.
# "opicFiltered" (renamed from "opic" on 2026-09-17, since a bare "opic" reads as a
# property of the data rather than as the filter that produced it) still tags outputs
# "_opic" -- the 143 existing outputs keep their names and nothing needs regenerating.
SUFFIX_ALIASES = {"opicFiltered": "opic"}
_TAG = re.sub(r"^summary_result_(noise1_)?", "", _STEM)
for _long, _short in SUFFIX_ALIASES.items():
    _TAG = _TAG.replace(_long, _short)
DATA_SUFFIX = "_" + _TAG


# --- packaged-input resolution ----------------------------------------------
# Inputs may sit either as full-width CSVs in the project root (the working copy) or
# as the slimmed, gzipped versions under data/ that package_inputs.py writes for the
# repository. pandas reads .csv.gz transparently, so resolving the name is all that is
# needed and no read site has to change. Search order:
#     <name>            data/<name>            <name>.gz            data/<name>.gz
# The LOGICAL name is what DATA_SUFFIX is derived from (above), so output names are
# identical either way -- a figure cannot reveal which copy produced it.
# An unresolvable name is returned unchanged, so the eventual read raises the normal
# FileNotFoundError naming the file the caller asked for.
DATA_DIR = os.environ.get("HR_DATA_DIR", "data")


def resolve_input(name):
    """Full-width CSV if present, else the packaged slim/gzipped copy."""
    for cand in (name, os.path.join(DATA_DIR, name), name + ".gz",
                 os.path.join(DATA_DIR, name + ".gz")):
        if os.path.isfile(cand):
            return cand
    return name


RESULTS_FILE = resolve_input(RESULTS_NAME)
# Topology resolves the same way, so a handoff carrying only data/ works: every
# builder below defaults to this rather than to the bare name.
NETWORK_JSON = resolve_input("network_data.json")

C_POS, C_NEG, C_MAIN = "#2196F3", "#F44336", "#3F51B5"

# --- network-pool switch (env HR_POOL) --------------------------------------
# The CANONICAL background for these statistics is the NON-DISJOINT (connected) pool:
# genuinely-disjoint networks (>= 2 weakly-connected components that each carry an edge;
# inactive edgeless genes don't count) must not sit in the reference/absent group. So the
# DEFAULT is "connected" (unsuffixed outputs). HR_POOL=full restores the full pool for
# comparison (outputs suffixed "_full"). Pool-aware scripts pass connected=CONNECTED.
POOL = os.environ.get("HR_POOL", "connected")
CONNECTED = (POOL != "full")
POOL_SUFFIX = "" if CONNECTED else "_full"

# The single suffix every reproduction script should append to its output names:
# results-file variant first, then pool (e.g. "_frac10", "_full", "_frac10_full").
OUT_SUFFIX = DATA_SUFFIX + POOL_SUFFIX

# --- second dynamics metric switch (env DYN_METRIC2) -------------------------
# The Phase 8-10 dynamics figures pair SETTLING TIME (top panel) with a second
# per-network metric (bottom panel). That second metric is pluggable:
#   traversal  (default) median PRE-SETTLING TRAVERSAL RATE -- total variation of
#                        each cell's trajectory over the pre-settling window, per
#                        unit expression level and per unit settling time
#   fluct                median pre-settling fluctuation, cv_before_x1 (tag "_fluct")
#   overshoot            fraction of the network's HR sets that overshoot,
#                        from frac_data/overshoot_results_net_*.csv
#   path / pathraw       diagnostics: traversal WITHOUT the per-time division, and
#                        without the expression-level normalisation
# The DEFAULT metric writes untagged names, so a bare run produces the canonical
# dyn_* figures; every other metric tags its outputs and cannot clobber them.
# Everything the overshoot run writes carries the extra tag "_overshoot" ahead of
# OUT_SUFFIX, so the fluctuation figures are never overwritten:
#   DYN_METRIC2=overshoot HR_RESULTS=summary_result_noise1_frac10.csv \
#       python reproduce_stability_dynamics.py    # -> dyn_*_overshoot_frac10.*
# --- noise-level set switch (env NOISE_SET) ----------------------------------
# The noise thread (Phases 15-19) compares three noise levels. There are two sets:
#   legacy  (default)  the original run: sigma 0.2 / 1.0 / 2.0, old schema files.
#                      Outputs unsuffixed, so the published nz_* stay put.
#   frac10             the 2026-08/09 simulations re-filtered at the 10% minimum
#                      state fraction: sigma 0.2 / 1.0 / 5.0. NOTE the high level
#                      is 5.0 here, NOT the legacy 2.0 -- the two sets' high arms
#                      are different experiments and must not be pooled.
#                      Outputs suffixed "_frac10".
# The noise scripts take the baseline (sigma 1.0) file of the SELECTED set as their
# topology/robustness source, so this switch alone is enough -- HR_RESULTS is not
# consulted and cannot desynchronise the two.
# Default changed from "legacy" to "frac10_opic" on 2026-09-17, so that a bare run of
# the noise thread matches the canonical HR_RESULTS default above. The legacy set
# (sigma 0.2 / 1.0 / 2.0, old schema) is still selectable and its high arm is a
# DIFFERENT experiment from the frac10 sets' sigma 5.0 -- never pool the two.
NOISE_SET = os.environ.get("NOISE_SET", "frac10_opic")
NOISE_SETS = {
    "legacy": dict(suffix="", levels=[0.2, 1.0, 2.0], files={
        0.2: "multimodal_results_combined_sorted_noise_0o2.csv",
        1.0: "multimodal_results_combined_sorted.csv",
        2.0: "multimodal_results_combined_sorted_noise_2.csv"}),
    "frac10": dict(suffix="_frac10", levels=[0.2, 1.0, 5.0], files={
        0.2: "summary_result_noise02_frac10.csv",
        1.0: "summary_result_noise1_frac10.csv",
        5.0: "summary_result_noise5_frac10.csv"}),
    "raw": dict(suffix="_raw5", levels=[0.2, 1.0, 5.0], files={
        0.2: "summary_result_noise02.csv",
        1.0: "summary_result_noise1_0817.csv",
        5.0: "summary_result_noise5.csv"}),
    # frac10 AND reproducible from opposite initial conditions (filter_op_ic_reproducible.py).
    # The op_ic requirement bites very unevenly: it removes 21% of sigma-0.2 sets but
    # almost nothing at sigma 5, and within sigma 0.2 it removes 76% of Pure Positive
    # sets against 1% of Pure Negative -- so this set is NOT a neutral tightening.
    "frac10_opic": dict(suffix="_frac10_opic", levels=[0.2, 1.0, 5.0], files={
        0.2: "summary_result_noise02_frac10_opicFiltered.csv",
        1.0: "summary_result_noise1_frac10_opicFiltered.csv",
        5.0: "summary_result_noise5_frac10_opicFiltered.csv"}),
}
_NZ = NOISE_SETS[NOISE_SET]
NOISE_LEVELS, NZ_SUFFIX = _NZ["levels"], _NZ["suffix"]
# resolved the same way as RESULTS_FILE, so the noise thread also runs off data/
NOISE_FILES = {lvl: resolve_input(p) for lvl, p in _NZ["files"].items()}

DYN_METRIC2 = os.environ.get("DYN_METRIC2", "traversal")
METRIC2 = {
    "fluct": dict(
        col="fluct_median", tag="_fluct", grad="fluctuation", pct=False,
        ols="dyn_fluct_ols", strat="dyn_robust_fluct_strat",
        bycat="dyn_fluct_by_category", bycat_col="median_fluct",
        desc="median cv_before_x1",
        axis="Pre-Settling Fluctuation (a.u.)",
        axis_median="Median Pre-Settling Fluctuation (a.u.)",
        perloop="Change in Median Fluctuation per Loop (a.u.)",
        forest="Spearman(Rate, Median Fluctuation)",
        within="Within-Network Spearman(Settling Time, Fluctuation)"),
    "overshoot": dict(
        col="over_rate", tag="_overshoot", grad="overshoot", pct=True,
        ols="dyn_overshoot_ols", strat="dyn_robust_overshoot_strat",
        bycat="dyn_overshoot_by_category", bycat_col="median_overshoot",
        desc="per-network overshoot fraction",
        axis="Overshoot Fraction of HR Sets",
        axis_median="Overshoot Fraction of HR Sets",
        perloop="Change in Overshoot Fraction per Loop",
        forest="Spearman(Rate, Overshoot Fraction)",
        within="Within-Network Spearman(Settling Time, Overshoot)"),
    # Single-cell trajectory PATH LENGTH, averaged over the accepted genes.
    # "path" divides each gene's path by that gene's own expression level, because
    # RAW path length is dominated by expression magnitude (Spearman +0.80 with
    # level, 75x spread across sets) and would otherwise measure scale rather than
    # motion. Normalised, it is scale-free like cv_before_x1 and correlates +0.55
    # with it. "pathraw" keeps the unnormalised version for comparison.
    "path": dict(
        col="path_median", tag="_path", grad="path", pct=False,
        ols="dyn_path_ols", strat="dyn_robust_path_strat",
        bycat="dyn_path_by_category", bycat_col="median_path",
        desc="median relative path length",
        axis="Path Length / Expression Level",
        axis_median="Median Path Length / Expression Level",
        perloop="Change in Median Relative Path Length per Loop",
        forest="Spearman(Rate, Median Relative Path Length)",
        within="Within-Network Spearman(Settling Time, Relative Path Length)"),
    # PRE-SETTLING TRAVERSAL RATE -- the adopted name for this metric.
    # Total variation of each cell's trajectory over the pre-settling window, divided
    # by the gene's expression level and by the settling time. Since each step moves
    # ~1.6% of the level, it approximates the mean absolute LOGARITHMIC rate of
    # change, <|d ln x / dt|>. Deliberately NOT called a speed or velocity: it is
    # unsigned total variation (a trajectory that oscillates back to where it began
    # still accumulates traversal), and it is per-gene L1, not the Euclidean speed of
    # the 3-gene state vector.
    "traversal": dict(
        col="path_median", tag="", grad="traversal", pct=False,
        ols="dyn_traversal_ols", strat="dyn_robust_traversal_strat",
        bycat="dyn_traversal_by_category", bycat_col="median_traversal",
        desc="median pre-settling traversal rate",
        axis="Pre-Settling Traversal Rate\n(per 1000 time units)",
        axis_median="Median Pre-Settling Traversal Rate\n(per 1000 time units)",
        perloop="Change in Traversal Rate per Loop\n(per 1000 time units)",
        forest="Spearman(Robustness, Traversal Rate)",
        within="Within-Network Spearman(Settling Time, Traversal Rate)"),
    "pathraw": dict(
        col="path_median", tag="_pathraw", grad="path", pct=False,
        ols="dyn_pathraw_ols", strat="dyn_robust_pathraw_strat",
        bycat="dyn_pathraw_by_category", bycat_col="median_path",
        desc="median raw path length",
        axis="Path Length (a.u.)",
        axis_median="Median Path Length (a.u.)",
        perloop="Change in Median Path Length per Loop (a.u.)",
        forest="Spearman(Rate, Median Path Length)",
        within="Within-Network Spearman(Settling Time, Path Length)"),
}[DYN_METRIC2]
M2_TAG = METRIC2["tag"]
M2_COL = METRIC2["col"]


def overshoot_per_network(results_file=None, min_runs_corr=20):
    """Per-network overshoot statistics, read straight from the results file.

    A set overshoots when the transient peak of the cell fraction exceeds its final
    value by the detector's threshold (ratio >= 1.1); the results file records that
    ratio in `overshoot_ratio`, which is NaN when there is no overshoot.

    Sourced from the results file rather than from frac_data since 2026-09-17. The
    two were verified equivalent on the sigma-1 frac10+op_ic set: over all 467,885
    rows the existence flag agrees 100.0000% with (on_ratio | off_ratio).notna(),
    and `overshoot_ratio` / `overshoot_time` / `overshoot_state` are numerically
    IDENTICAL to frac_data's ratio / peak time / side. frac_data additionally holds
    10,817 sets that the frac10 and op_ic filters removed -- correctly absent here --
    and its early/late label, which is exactly `overshoot_time > 1000`. Reading the
    results file therefore loses nothing, drops a 32 MB second input, and removes the
    (net_id, param_id, accepted_nodes) merge as a failure mode.

    Returns net_id, n_sets, n_over, over_rate, and `over_corr` -- the within-network
    Spearman(stability_time, overshoot) over networks with >= min_runs_corr sets.
    """
    frac = pd.read_csv(results_file or RESULTS_FILE,
                       usecols=["net_id", "stability_time", "overshoot_ratio"])
    frac["overshoot"] = frac.overshoot_ratio.notna().astype(int)

    agg = (frac.groupby("net_id")
               .agg(n_sets=("overshoot", "size"), n_over=("overshoot", "sum"))
               .reset_index())
    agg["over_rate"] = agg.n_over / agg.n_sets

    def _corr(g):
        if len(g) < min_runs_corr or g.overshoot.nunique() < 2:
            return np.nan
        return stats.spearmanr(g.stability_time, g.overshoot).correlation
    agg = agg.merge(frac.groupby("net_id")[["stability_time", "overshoot"]]
                        .apply(_corr).rename("over_corr").reset_index(),
                    on="net_id", how="left")
    return agg

PATH_RATE_SCALE = 1000.0     # see per_time below


def path_per_network(results_file=None, normalize=True, per_time=False,
                     min_settling=10.0, min_runs_corr=20):
    """Per-network single-cell trajectory PATH LENGTH from the results file.

    Each set carries `mean_path_x{1,2,3}` -- the mean over the 500 cells of the arc
    length of that gene's trajectory. Only the ACCEPTED genes are used: which genes
    the GMM accepted varies per set, and the three genes' path lengths are nearly
    independent (mean cross-gene Spearman 0.13), so a fixed gene would be a lottery.

    normalize=True divides each gene's path by that gene's own expression level
    (mean of its two GMM component means), making the metric scale-free; raw path
    length tracks expression magnitude at Spearman +0.80 and would otherwise
    dominate any topology comparison.

    per_time=True further divides by that set's settling time, giving the
    PRE-SETTLING TRAVERSAL RATE. Total path necessarily grows with the settling
    window it accumulates over (rho +0.53), so dividing by that window is not a
    rescaling but the removal of a built-in confound. Sets settling in under `min_settling` time units are dropped
    from the rate metric -- 73 of 471k at sigma 1.0 -- since dividing by a near-zero
    settling time yields speeds ~1000x the typical value.

    Returns net_id, n_sets, path_median/mean, and `path_corr` -- the within-network
    Spearman(stability_time, path) over networks with >= min_runs_corr sets.
    """
    genes = ["x1", "x2", "x3"]
    cols = (["net_id", "param_id", "accepted_nodes", "stability_time"]
            + [f"mean_path_{g}" for g in genes]
            + [f"gmm_mean{i}_{g}" for g in genes for i in (1, 2)])
    d = pd.read_csv(results_file or RESULTS_FILE, usecols=cols)

    P = np.stack([d[f"mean_path_{g}"].values for g in genes], axis=1)
    if normalize:
        lev = np.stack([d[[f"gmm_mean1_{g}", f"gmm_mean2_{g}"]].mean(axis=1).values
                        for g in genes], axis=1)
        P = np.where(lev > 1e-9, P / lev, np.nan)
    acc = np.stack([d.accepted_nodes.str.contains(g).values for g in genes], axis=1)
    with np.errstate(invalid="ignore"):
        d["path"] = np.nanmean(np.where(acc, P, np.nan), axis=1)
    if per_time:
        # x1000 so the metric reads "path per 1000 time units": the raw per-unit-time
        # value is ~0.0155 and per-loop OLS coefficients would print as 0.000
        d["path"] = np.where(d.stability_time >= min_settling,
                             PATH_RATE_SCALE * d.path / d.stability_time, np.nan)
    d = d.dropna(subset=["path"])

    agg = (d.groupby("net_id")
             .agg(n_sets=("path", "size"), path_median=("path", "median"),
                  path_mean=("path", "mean"))
             .reset_index())

    def _corr(g):
        if len(g) < min_runs_corr:
            return np.nan
        return stats.spearmanr(g.stability_time, g.path).correlation
    agg = agg.merge(d.groupby("net_id")[["stability_time", "path"]]
                     .apply(_corr).rename("path_corr").reset_index(),
                    on="net_id", how="left")
    return agg


PRETTY = {
    "fb_pos_1node": "Pos 1-node (self)", "fb_neg_1node": "Neg 1-node (self)",
    "fb_pos_2node": "Pos 2-node",         "fb_neg_2node": "Neg 2-node",
    "fb_pos_3node": "Pos 3-node",         "fb_neg_3node": "Neg 3-node",
    "fb_pos_feedback": "Any positive feedback", "fb_neg_feedback": "Any negative feedback",
}


def savefig(path_noext, **kw):
    """Save current matplotlib figure as both PNG and SVG (project convention)."""
    import matplotlib.pyplot as plt
    plt.savefig(path_noext + ".png", dpi=150, **kw)
    plt.savefig(path_noext + ".svg", **kw)
    plt.close()


# =============================================================================
# Data assembly — everything from the two raw files
# =============================================================================
def load_adjacency_matrices(network_json=None):
    # None -> NETWORK_JSON, which resolve_input has already pointed at either the
    # project-root copy or data/network_data.json. build_network_table,
    # build_feature_table and build_dynamics_table all funnel through here, so this
    # is the single place the topology path is decided.
    network_json = network_json or NETWORK_JSON
    return {e["unique_id"]: np.array(e["adj_matrix"])
            for e in json.load(open(network_json))}


def _read_result_netids(results_file):
    if str(results_file).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(results_file, usecols=["net_id"])["net_id"]
    return pd.read_csv(results_file, usecols=["net_id"])["net_id"]


def edge_bearing_components(mat):
    """Number of weakly-connected components that contain >= 1 edge — the refined
    'disjoint' metric. A component with an off-diagonal edge counts; a lone node counts
    only if it carries a self-loop; genes with no incident edge (inactive / not modelled)
    are ignored. A network is DISJOINT when this is >= 2 (e.g. two separate self-loops),
    but NOT when it is a single edge-bearing component plus some edgeless genes."""
    n = mat.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] != 0 or mat[j, i] != 0:
                G.add_edge(i, j)
    return sum(1 for c in nx.connected_components(G)
               if len(c) >= 2 or any(mat[i, i] != 0 for i in c))


def active_nodes(mat):
    """1-indexed IDs of nodes carrying >= 1 incident edge (a self-loop OR any
    off-diagonal in/out edge). Inactive genes (no incident edge) are not modelled and
    are omitted — most networks have all 3 active, some have only 1 or 2."""
    n = mat.shape[0]
    return [i + 1 for i in range(n) if mat[i, :].any() or mat[:, i].any()]


def build_network_table(network_json=None,
                        results_file=RESULTS_FILE,
                        n_trials=N_TRIALS_DEFAULT, connected=False):
    """One row per network with everything the reproductions need:
       label (>=1 multimodal run), multimodal_count, multimodal_rate, n_edges,
       full feedback taxonomy counts (fb_*) + presence flags (has_fb_*),
       feedforward flags (has_ffl_*), and polarity category.
       connected=True drops genuinely-disjoint networks (edge_bearing_components >= 2).
    """
    mats = load_adjacency_matrices(network_json)
    counts = _read_result_netids(results_file).value_counts()   # successful sets/net
    rows = []
    for uid, m in mats.items():
        if connected and edge_bearing_components(m) >= 2:
            continue
        fb, ff = feedback_counts(m), feedforward_counts(m)
        cnt = int(counts.get(uid, 0))
        rec = {"net_id": uid, "n_edges": int(np.sum(m != 0)),
               "multimodal_count": cnt, "label": int(cnt > 0),
               "multimodal_rate": cnt / n_trials}
        for cat in FEEDBACK_CATEGORIES + ["pos_feedback", "neg_feedback", "total_feedback"]:
            rec[f"fb_{cat}"] = fb[cat]
            rec[f"has_fb_{cat}"] = fb[f"has_{cat}"]
        for k in ("ffl_coherent", "ffl_incoherent", "ffl_total",
                  "has_ffl_coherent", "has_ffl_incoherent", "has_ffl_total"):
            rec[k] = ff[k]
        rows.append(rec)
    df = pd.DataFrame(rows)
    pos, neg = df["has_fb_pos_feedback"] == 1, df["has_fb_neg_feedback"] == 1
    df["category"] = np.select([pos & ~neg, neg & ~pos, pos & neg],
                               ["PFL-only", "NFL-only", "Mixed"], default="No feedback")
    return df


# =============================================================================
# Full topology feature table (superset of build_network_table)
# =============================================================================
# Reproduces results/features.csv column-for-column from ONLY the two raw files,
# so ML analyses that need the full 63-feature topology set (Phases 20-22) can run
# with no dependence on the derived features.csv. Feature logic mirrors
# 1_feature_engineering.py exactly (same column names AND order -> byte-identical
# feature matrix, which keeps seeded RandomForest results reproducible).
def _signed_cycles(mat, length):
    """(#positive, #negative) simple cycles of given length in a signed digraph.
    A cycle is positive/negative by the product of its edge signs."""
    n = mat.shape[0]
    n_pos = n_neg = 0
    for cyc in itertools.permutations(range(n), length):
        sign, valid = 1, True
        for i in range(length):
            w = mat[cyc[i], cyc[(i + 1) % length]]
            if w == 0:
                valid = False
                break
            sign *= w
        if not valid:
            continue
        if sign == 1:
            n_pos += 1
        else:
            n_neg += 1
    return n_pos // length, n_neg // length   # each undirected cycle counted `length` times


def _build_digraph(mat):
    G = nx.DiGraph()
    n = mat.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if mat[i, j] != 0:
                G.add_edge(i, j, sign=int(mat[i, j]))
    return G


def _extract_features(uid, mat):
    """All topological features for one network (mirrors 1_feature_engineering.py)."""
    n = mat.shape[0]
    f = {"net_id": uid}

    # A. Raw structure
    for i in range(n):
        f[f"self_reg_{i+1}"] = int(mat[i, i])
    for i in range(n):
        for j in range(n):
            if i != j:
                f[f"edge_{i+1}{j+1}"] = int(mat[i, j])
    f["n_activating"] = int(np.sum(mat == 1))
    f["n_repressing"] = int(np.sum(mat == -1))
    f["n_zero"]       = int(np.sum(mat == 0))
    f["n_edges"]      = int(np.sum(mat != 0))
    f["edge_density"] = f["n_edges"] / (n * n)
    diag = np.diag(mat)
    f["n_self_activating"] = int(np.sum(diag == 1))
    f["n_self_repressing"] = int(np.sum(diag == -1))
    f["n_self_none"]       = int(np.sum(diag == 0))
    off = mat[~np.eye(n, dtype=bool)]
    f["n_off_activating"] = int(np.sum(off == 1))
    f["n_off_repressing"] = int(np.sum(off == -1))
    f["n_off_zero"]       = int(np.sum(off == 0))

    # B. Degree (self-regulation excluded)
    for i in range(n):
        in_off  = np.array([mat[j, i] for j in range(n) if j != i])
        out_off = np.array([mat[i, j] for j in range(n) if j != i])
        f[f"in_deg_pos_{i+1}"]   = int(np.sum(in_off == 1))
        f[f"in_deg_neg_{i+1}"]   = int(np.sum(in_off == -1))
        f[f"out_deg_pos_{i+1}"]  = int(np.sum(out_off == 1))
        f[f"out_deg_neg_{i+1}"]  = int(np.sum(out_off == -1))
        f[f"in_deg_total_{i+1}"] = int(np.sum(in_off != 0))
        f[f"out_deg_total_{i+1}"]= int(np.sum(out_off != 0))
        f[f"net_in_deg_{i+1}"]   = f[f"in_deg_pos_{i+1}"] - f[f"in_deg_neg_{i+1}"]
    in_tot  = [f[f"in_deg_total_{i+1}"]  for i in range(n)]
    out_tot = [f[f"out_deg_total_{i+1}"] for i in range(n)]
    f["max_in_degree"]  = max(in_tot)
    f["max_out_degree"] = max(out_tot)
    f["mean_in_degree"] = np.mean(in_tot)
    f["mean_out_degree"]= np.mean(out_tot)

    # C. Cycle counts (deprecated set, retained for column parity) + feedback taxonomy
    pos2, neg2 = _signed_cycles(mat, 2)
    pos3, neg3 = _signed_cycles(mat, 3)
    f["n_pos_2cycles"], f["n_neg_2cycles"], f["n_2cycles"] = pos2, neg2, pos2 + neg2
    f["n_pos_3cycles"], f["n_neg_3cycles"], f["n_3cycles"] = pos3, neg3, pos3 + neg3
    f["n_pos_cycles_total"] = pos2 + pos3
    f["n_neg_cycles_total"] = neg2 + neg3
    f["n_cycles_total"]     = pos2 + neg2 + pos3 + neg3
    f["has_pos_cycle"] = int((pos2 + pos3) > 0)
    f["has_neg_cycle"] = int((neg2 + neg3) > 0)
    f["has_any_cycle"] = int((pos2 + neg2 + pos3 + neg3) > 0)
    f["has_self_activation"] = int(np.any(diag == 1))
    f["has_self_repression"] = int(np.any(diag == -1))
    fb = feedback_counts(mat)
    for cat in ["pos_1node", "neg_1node", "pos_2node", "neg_2node", "pos_3node",
                "neg_3node", "pos_feedback", "neg_feedback", "total_feedback"]:
        f[f"fb_{cat}"]     = fb[cat]
        f[f"has_fb_{cat}"] = fb[f"has_{cat}"]

    # D. Connectivity (unsigned digraph)
    G = _build_digraph(mat)
    sccs = list(nx.strongly_connected_components(G))
    f["n_scc"]            = len(sccs)
    f["largest_scc_size"] = max(len(s) for s in sccs)
    f["is_strongly_connected"] = int(nx.is_strongly_connected(G))
    f["n_wcc"] = len(list(nx.weakly_connected_components(G)))
    f["is_weakly_connected"] = int(nx.is_weakly_connected(G))
    f["reachability"] = sum(1 for i in range(n) for j in range(n)
                            if i != j and nx.has_path(G, i, j)) / (n * (n - 1))

    # E. Spectral (raw signed matrix)
    ev = np.linalg.eigvals(mat.astype(float))
    re = ev.real
    f["trace"]               = float(np.trace(mat))
    f["det"]                 = float(np.linalg.det(mat.astype(float)))
    f["spectral_radius"]     = float(np.max(np.abs(ev)))
    f["max_eigenvalue_real"] = float(np.max(re))
    f["min_eigenvalue_real"] = float(np.min(re))
    f["sum_eigenvalue_real"] = float(np.sum(re))
    f["eigenvalue_spread"]   = float(np.max(re) - np.min(re))
    f["frobenius_norm"]      = float(np.linalg.norm(mat.astype(float), "fro"))
    return f


def build_feature_table(network_json=None,
                        results_file=RESULTS_FILE,
                        n_trials=N_TRIALS_DEFAULT, connected=False):
    """Full per-network topology feature table built ONLY from the two raw files —
    reproduces results/features.csv (same columns AND order): raw structure, degree,
    cycle counts, feedback taxonomy, connectivity, spectral, plus label /
    multimodal_count / multimodal_rate and the threshold labels (t05/t10/t25).
    connected=True drops genuinely-disjoint networks (edge_bearing_components >= 2)."""
    mats = load_adjacency_matrices(network_json)
    counts = _read_result_netids(results_file).value_counts()
    feats = []
    for uid, m in mats.items():
        mm = np.array(m)
        if connected and edge_bearing_components(mm) >= 2:
            continue
        feats.append(_extract_features(uid, mm))
    df = pd.DataFrame(feats)
    df["label"]           = df["net_id"].map(lambda u: int(counts.get(u, 0) > 0))
    df["multimodal_count"] = df["net_id"].map(lambda u: int(counts.get(u, 0)))
    df["multimodal_rate"]  = df["multimodal_count"] / n_trials
    for t in (0.05, 0.10, 0.25):
        df[f"label_t{int(t*100):02d}"] = (df["multimodal_rate"] >= t).astype(int)
    return df


# =============================================================================
# Dynamics table (Phases 8-10: settling time & pre-stability fluctuations)
# =============================================================================
def spearman_ci(r, n, alpha=0.05):
    """Approximate 95% CI for a Spearman correlation via the Fisher z-transform."""
    if n is None or n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return (float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se)))


# The adopted 5-way feedback gradient (positive -> negative). Used across the
# dynamics and noise reproductions.
GROUP5 = ["Pure Positive", "Mixed Pos-Dom", "Balanced", "Mixed Neg-Dom", "Pure Negative"]


def feedback_group5(df):
    """5-way feedback grouping = polarity `category` x dominance. Splits the broad
    "Mixed" (has-both) group into Pos-Dom / Balanced / Neg-Dom by the sign of
    (fb_pos_feedback - fb_neg_feedback). Requires either a `category` column or the
    `has_fb_pos_feedback` / `has_fb_neg_feedback` flags, plus the two count columns.
    Returns a numpy array (No-feedback networks -> "No feedback", filter them out)."""
    if "category" in df.columns:
        cat = df["category"].values
    else:
        p, n = df["has_fb_pos_feedback"] == 1, df["has_fb_neg_feedback"] == 1
        cat = np.select([p & ~n, n & ~p, p & n], ["PFL-only", "NFL-only", "Mixed"],
                        default="No feedback")
    d = (df["fb_pos_feedback"] - df["fb_neg_feedback"]).values
    return np.select(
        [cat == "No feedback", cat == "PFL-only", cat == "NFL-only", d > 0, d < 0],
        ["No feedback", "Pure Positive", "Pure Negative", "Mixed Pos-Dom", "Mixed Neg-Dom"],
        default="Balanced")


def build_dynamics_table(network_json=None,
                         results_file=RESULTS_FILE,
                         feature_df=None, min_runs_corr=20, fluct_col="cv_before_x1"):
    """Per-network DYNAMICS table for Phases 8-10, from the raw results file:
    settling time (`stability_time`) and a fluctuation metric (`fluct_col`, default
    the pre-stability `cv_before_x1`; pass `cv_after_x1` for post-settling) aggregated
    per network, merged with the topology needed for the category systems. Scope =
    multimodal networks (multimodal_rate > 0).

    Columns: n_runs, st_median/mean/std/cv (settling time), fluct_median/mean
    (median/mean of `fluct_col`), within_corr (within-network Spearman(stability_time,
    fluct_col) over that network's parameter sets, NaN if < min_runs_corr),
    multimodal_rate, n_edges, largest_scc_size, feedback counts + polarity
    `category` (PFL-only / NFL-only / Mixed), and the Phase 9/10 category systems
    (rate_q, scc_profile, density_bin, polarity_class)."""
    runs = pd.read_csv(results_file, usecols=["net_id", "stability_time", fluct_col])
    g = runs.groupby("net_id")
    net = g.agg(n_runs=("stability_time", "size"),
                st_median=("stability_time", "median"),
                st_mean=("stability_time", "mean"),
                st_std=("stability_time", "std"),
                fluct_median=(fluct_col, "median"),
                fluct_mean=(fluct_col, "mean")).reset_index()
    net["st_cv"] = net["st_std"] / net["st_mean"]

    def _within(sub):
        s = sub.dropna()
        if (len(s) < min_runs_corr or s[fluct_col].nunique() < 3
                or s["stability_time"].nunique() < 3):
            return np.nan
        return stats.spearmanr(s["stability_time"], s[fluct_col]).correlation
    wc = g[["stability_time", fluct_col]].apply(_within).rename("within_corr")
    net = net.merge(wc, on="net_id", how="left")

    # topology: full feature table has largest_scc_size (needed for scc_profile);
    # derive polarity `category` inline if the supplied table lacks it.
    feats = (feature_df if feature_df is not None
             else build_feature_table(network_json, results_file)).copy()
    if "category" not in feats.columns:
        pos, neg = feats["has_fb_pos_feedback"] == 1, feats["has_fb_neg_feedback"] == 1
        feats["category"] = np.select([pos & ~neg, neg & ~pos, pos & neg],
                                      ["PFL-only", "NFL-only", "Mixed"], default="No feedback")
    keep = ["net_id", "multimodal_rate", "n_edges", "largest_scc_size",
            "fb_pos_feedback", "fb_neg_feedback", "fb_total_feedback", "category"]
    net = net.merge(feats[keep], on="net_id", how="left")
    net = net[net["multimodal_rate"] > 0].reset_index(drop=True)

    # --- category systems (identical definitions to Phases 9 & 10) -----------
    net["rate_q"] = pd.qcut(net["multimodal_rate"], 4,
                            labels=["Q1 (least robust)", "Q2", "Q3", "Q4 (most robust)"])
    net["scc_profile"] = net["largest_scc_size"].map(
        lambda s: "(3) strongly conn." if s == 3 else "(2,1)" if s == 2 else "(1,1,1)")
    net["density_bin"] = pd.cut(net["n_edges"], bins=[-np.inf, 4, 5, 6, np.inf],
                                labels=["<=4 edges", "5 edges", "6 edges", ">=7 edges"])
    d = net["fb_pos_feedback"] - net["fb_neg_feedback"]
    net["polarity_class"] = np.where(d > 0, "pos-dominant",
                            np.where(d < 0, "neg-dominant", "balanced"))
    net["fb_group5"] = feedback_group5(net)     # 5-way gradient (see helper)
    return net


# =============================================================================
# Statistic 1 — density-controlled odds ratio (identity / binary multimodality)
# =============================================================================
def crude_and_mh(df, presence_flag, outcome="label", strata="n_edges"):
    """Crude log-OR (Haldane-corrected) + density-controlled Mantel-Haenszel
    log-OR with Cochran-Mantel-Haenszel test and Breslow-Day homogeneity.
    Strata = distinct values of `strata`; degenerate strata (no exposure or no
    outcome variation) are dropped. Returns a flat dict (MH fields are NaN when
    no stratum is informative, e.g. 3-node feedback)."""
    y = df[outcome].values
    crude = motif_contingency(df[presence_flag].values, y)
    out = {"presence_flag": presence_flag, "n_present": crude["n_present"],
           "pct_present": round(crude["pct_pos_when_present"], 1),
           "pct_absent": round(crude["pct_pos_when_absent"], 1),
           "crude_log_or": round(crude["log_or"], 3),
           "crude_ci_low": round(np.log(crude["ci_low"]), 3),
           "crude_ci_high": round(np.log(crude["ci_high"]), 3),
           "haldane_corrected": crude["corrected"]}

    tables, used = [], []
    for lvl, sub in df.groupby(strata):
        p = sub[presence_flag].values.astype(bool)
        yv = sub[outcome].values.astype(int)
        a = int((p & (yv == 1)).sum()); b = int((p & (yv == 0)).sum())
        c = int((~p & (yv == 1)).sum()); d = int((~p & (yv == 0)).sum())
        if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue
        tables.append([[a, b], [c, d]]); used.append(int(lvl))

    base = {"mh_log_or": np.nan, "mh_ci_low": np.nan, "mh_ci_high": np.nan,
            "cmh_chi2": np.nan, "cmh_p": np.nan, "breslow_day_p": np.nan,
            "n_strata": len(used), "strata": "+".join(map(str, used)), "mh_note": ""}
    if not tables:
        base.update(n_strata=0, mh_note="no informative strata")
        out.update(base); return out

    st = StratifiedTable(np.array(tables).transpose(1, 2, 0))
    or_pooled = st.oddsratio_pooled
    if not np.isfinite(or_pooled) or or_pooled <= 0:
        # e.g. 3-node feedback: present group is 100% multimodal in every stratum
        # (zero within-stratum discordance) -> MH OR infinite / not identifiable.
        base["mh_note"] = "not identifiable (present group multimodal in every stratum)"
        out.update(base); return out

    cmh = st.test_null_odds(correction=True)
    lo, hi = st.oddsratio_pooled_confint()
    try:
        bd_p = round(st.test_equal_odds().pvalue, 3)
    except Exception:
        bd_p = np.nan
    base.update({"mh_log_or": round(np.log(or_pooled), 3),
                 "mh_ci_low": round(np.log(lo), 3), "mh_ci_high": round(np.log(hi), 3),
                 "cmh_chi2": round(cmh.statistic, 3), "cmh_p": cmh.pvalue,
                 "breslow_day_p": bd_p})
    out.update(base)
    return out


# =============================================================================
# Statistic 2 — binomial GLM per-loop log-odds (robustness / multimodal rate)
# =============================================================================
def binomial_glm_per_loop(df, predictor_cols, count_col="multimodal_count",
                          n_trials=N_TRIALS_DEFAULT):
    """Binomial GLM: multimodal_count ~ Binomial(n_trials, p), logit(p)=X.beta.
    Coefficients are log odds ratios per ADDITIONAL loop of each type, on the
    per-parameter-set probability of multimodality. Quasi-binomial CIs inflate
    SEs by sqrt(dispersion) to honour network-level overdispersion (point
    estimates unchanged). Returns a DataFrame (incl. const)."""
    endog = np.column_stack([df[count_col].values, n_trials - df[count_col].values])
    X = sm.add_constant(df[predictor_cols].astype(float))
    fit = sm.GLM(endog, X, family=sm.families.Binomial()).fit()
    disp = fit.pearson_chi2 / fit.df_resid
    beta = fit.params.values
    qse = fit.bse.values * np.sqrt(disp)
    z = beta / qse
    pval = 2 * stats.norm.sf(np.abs(z))
    out = pd.DataFrame({
        "feature": ["const"] + list(predictor_cols),
        "log_or": beta, "or": np.exp(beta),
        "ci_low": np.exp(beta - 1.96 * qse), "ci_high": np.exp(beta + 1.96 * qse),
        "log_or_ci_low": beta - 1.96 * qse, "log_or_ci_high": beta + 1.96 * qse,
        "pval": pval,
    })
    out.attrs["dispersion"] = disp
    out.attrs["params"] = fit.params               # for glm_contrast
    out.attrs["quasi_cov"] = fit.cov_params() * disp
    return out


def glm_contrast(glm_df, feat_a, feat_b):
    """Quasi-binomial Wald test of (coef[feat_a] - coef[feat_b]) = 0, using the fit
    stored on a binomial_glm_per_loop() result. Returns the log-OR difference, its
    95% CI and p-value."""
    p, cov = glm_df.attrs["params"], glm_df.attrs["quasi_cov"]
    diff = float(p[feat_a] - p[feat_b])
    var = float(cov.loc[feat_a, feat_a] + cov.loc[feat_b, feat_b]
                - 2 * cov.loc[feat_a, feat_b])
    se = np.sqrt(var)
    z = diff / se
    return {"diff_log_or": diff, "se": se, "ci_low": diff - 1.96 * se,
            "ci_high": diff + 1.96 * se, "z": z, "pval": 2 * stats.norm.sf(abs(z))}
