"""
motif_lib.py — Standardized terminology & helpers for GRN topology analysis.

============================================================================
STANDARDIZED FEEDBACK TAXONOMY
============================================================================
A feedback loop is a directed cycle. Its SIGN is the product of the signs of
its edges:
    positive feedback (pos)  = sign product +1   (reinforcing)
    negative feedback (neg)  = sign product -1   (balancing)

Self-loops (diagonal entries) are included as feedback. We distinguish three
SCALES by the number of nodes in the loop:

    1-node feedback (self)  : a single diagonal entry
                              pos = self-activation (+1)
                              neg = self-repression (-1)
    2-node feedback         : a mutual regulation i<->j, sign = s_ij * s_ji
                              pos = (++ or --)  product +1
                              neg = (+-)         product -1
    3-node feedback         : a directed 3-cycle, sign = product of 3 edges
                              pos = 0 or 2 repressive edges (product +1)
                              neg = 1 or 3 repressive edges (product -1)

This yields 2 polarities x 3 scales = 6 elementary feedback categories:
    pos_1node, neg_1node, pos_2node, neg_2node, pos_3node, neg_3node

(The older term "2-cycle"/"3-cycle" is deprecated in favour of
 "2-node feedback"/"3-node feedback".)

============================================================================
MOTIF ISOMORPHISM
============================================================================
Two signed 3-node networks are ISOMORPHIC if one can be obtained from the
other by relabelling nodes (a permutation in S_3 acting on the matrix as
A -> P A P^T). The canonical form is the lexicographically smallest matrix
over all 6 permutations; this serves as the motif-class identifier. All
feedback-taxonomy counts defined above are isomorphism-invariant.
============================================================================
"""

import itertools
import numpy as np

# ---------------------------------------------------------------------------
# Feedback taxonomy
# ---------------------------------------------------------------------------
FEEDBACK_CATEGORIES = [
    "pos_1node", "neg_1node",
    "pos_2node", "neg_2node",
    "pos_3node", "neg_3node",
]

SCALES = ["1node", "2node", "3node"]
POLARITIES = ["pos", "neg"]

# The 6 elementary + 3 aggregate count columns and their presence flags, as they
# appear in results/features.csv (written by 1_feature_engineering.py).
FB_COUNT_COLS = [f"fb_{c}" for c in FEEDBACK_CATEGORIES] + \
                ["fb_pos_feedback", "fb_neg_feedback", "fb_total_feedback"]
FB_FLAG_COLS = ["has_" + c for c in FB_COUNT_COLS]

# Deprecated "2-cycle / 3-cycle" feature columns, superseded by the fb_* taxonomy.
# They remain in features.csv for backward compatibility but should be EXCLUDED
# from standardized figures / modeling so the fb_* naming is the single source.
DEPRECATED_FEATURE_COLS = [
    "n_pos_2cycles", "n_neg_2cycles", "n_2cycles",
    "n_pos_3cycles", "n_neg_3cycles", "n_3cycles",
    "n_pos_cycles_total", "n_neg_cycles_total", "n_cycles_total",
    "has_pos_cycle", "has_neg_cycle", "has_any_cycle",
    "has_self_activation", "has_self_repression",
]

# Human-readable labels for the standardized feedback features (for plots).
FB_PRETTY = {
    "fb_pos_1node": "Pos 1-node (self)",  "fb_neg_1node": "Neg 1-node (self)",
    "fb_pos_2node": "Pos 2-node",          "fb_neg_2node": "Neg 2-node",
    "fb_pos_3node": "Pos 3-node",          "fb_neg_3node": "Neg 3-node",
    "fb_pos_feedback": "Any pos feedback", "fb_neg_feedback": "Any neg feedback",
    "fb_total_feedback": "Any feedback",
    "has_fb_pos_1node": "Pos 1-node (self)",  "has_fb_neg_1node": "Neg 1-node (self)",
    "has_fb_pos_2node": "Pos 2-node",          "has_fb_neg_2node": "Neg 2-node",
    "has_fb_pos_3node": "Pos 3-node",          "has_fb_neg_3node": "Neg 3-node",
}


def feedback_counts(mat):
    """
    Count feedback loops of every taxonomy category in a signed adjacency
    matrix (n x n, entries in {-1,0,1}). Returns a dict with the 6 elementary
    counts plus convenient aggregates.
    """
    n = mat.shape[0]
    c = {k: 0 for k in FEEDBACK_CATEGORIES}

    # 1-node (self-loops): diagonal entries
    diag = np.diag(mat)
    c["pos_1node"] = int(np.sum(diag == 1))
    c["neg_1node"] = int(np.sum(diag == -1))

    # 2-node feedback: unordered pairs i<j with both directed edges present
    for i, j in itertools.combinations(range(n), 2):
        s_ij, s_ji = mat[i, j], mat[j, i]
        if s_ij == 0 or s_ji == 0:
            continue
        if s_ij * s_ji == 1:
            c["pos_2node"] += 1
        else:
            c["neg_2node"] += 1

    # 3-node feedback: directed 3-cycles (each counted once)
    for trio in itertools.combinations(range(n), 3):
        for i, j, k in itertools.permutations(trio):
            # fix canonical rotation start to avoid triple counting:
            if i != min(trio):
                continue
            s1, s2, s3 = mat[i, j], mat[j, k], mat[k, i]
            if s1 == 0 or s2 == 0 or s3 == 0:
                continue
            if s1 * s2 * s3 == 1:
                c["pos_3node"] += 1
            else:
                c["neg_3node"] += 1

    # Aggregates
    c["pos_feedback"] = c["pos_1node"] + c["pos_2node"] + c["pos_3node"]
    c["neg_feedback"] = c["neg_1node"] + c["neg_2node"] + c["neg_3node"]
    c["total_feedback"] = c["pos_feedback"] + c["neg_feedback"]
    for cat in FEEDBACK_CATEGORIES + ["pos_feedback", "neg_feedback", "total_feedback"]:
        c[f"has_{cat}"] = int(c[cat] > 0)
    return c


# ---------------------------------------------------------------------------
# Feedforward loops (FFL) — distinct from feedback: acyclic, no directed cycle
# ---------------------------------------------------------------------------
# A 3-node feedforward loop has a SOURCE X with a direct edge X->Z and an
# indirect path X->Y->Z (Y = intermediate, Z = target). It is a DAG, not a
# cycle, so it is structurally distinct from the feedback loops above.
#
# Signed FFL coherence (Alon's definition):
#     direct sign  = s(X->Z)
#     indirect sign = s(X->Y) * s(Y->Z)
#     COHERENT   : direct == indirect  <=>  s(X->Y)*s(Y->Z)*s(X->Z) = +1
#     INCOHERENT : direct != indirect  <=>  s(X->Y)*s(Y->Z)*s(X->Z) = -1
#
# Each FFL is identified by an ordered (source, intermediate, target) role
# assignment; reverse edges (if any) are irrelevant to the feedforward pattern
# but may independently create feedback — control for feedback separately when
# isolating an FFL effect.
FFL_CATEGORIES = ["coherent", "incoherent"]

FFL_PRETTY = {
    "ffl_coherent": "Coherent FFL", "ffl_incoherent": "Incoherent FFL",
    "ffl_total": "Any FFL",
    "has_ffl_coherent": "Coherent FFL", "has_ffl_incoherent": "Incoherent FFL",
    "has_ffl_total": "Any FFL",
}


def feedforward_counts(mat):
    """
    Count signed feedforward loops in a signed adjacency matrix (entries in
    {-1,0,1}). For every ordered triple (X=source, Y=intermediate, Z=target)
    of distinct nodes with all three edges X->Y, Y->Z, X->Z present, classify
    as coherent (edge-sign product +1) or incoherent (-1).

    Returns dict with counts (ffl_coherent, ffl_incoherent, ffl_total) and the
    matching presence flags (has_ffl_*).
    """
    n = mat.shape[0]
    coh = inc = 0
    for X, Y, Z in itertools.permutations(range(n), 3):
        s_xy, s_yz, s_xz = mat[X, Y], mat[Y, Z], mat[X, Z]
        if s_xy == 0 or s_yz == 0 or s_xz == 0:
            continue
        if s_xy * s_yz * s_xz == 1:
            coh += 1
        else:
            inc += 1
    total = coh + inc
    return {
        "ffl_coherent": coh, "ffl_incoherent": inc, "ffl_total": total,
        "has_ffl_coherent": int(coh > 0), "has_ffl_incoherent": int(inc > 0),
        "has_ffl_total": int(total > 0),
    }


# ---------------------------------------------------------------------------
# Motif isomorphism
# ---------------------------------------------------------------------------
def canonical_form(mat):
    """
    Return the canonical (lexicographically smallest) flattened tuple over all
    node relabellings (S_n acting as A -> P A P^T). Identifies the motif class.
    """
    n = mat.shape[0]
    best = None
    for perm in itertools.permutations(range(n)):
        p = np.array(perm)
        permuted = mat[np.ix_(p, p)]
        flat = tuple(int(x) for x in permuted.flatten())
        if best is None or flat < best:
            best = flat
    return best


def n_automorphisms(mat):
    """Number of node permutations that leave the matrix unchanged (orbit symmetry)."""
    n = mat.shape[0]
    base = tuple(int(x) for x in mat.flatten())
    count = 0
    for perm in itertools.permutations(range(n)):
        p = np.array(perm)
        if tuple(int(x) for x in mat[np.ix_(p, p)].flatten()) == base:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Log odds ratio (with Haldane-Anscombe correction for zero cells)
# ---------------------------------------------------------------------------
def log_odds_ratio(a, b, c, d, correction=0.5):
    """
    Odds ratio for the association between a binary motif (present/absent) and
    a binary outcome (e.g. multimodal=1 / non-multimodal=0).

    Contingency layout:
                       outcome=1   outcome=0
        motif present      a           b
        motif absent       c           d

    OR = odds(outcome=1 | present) / odds(outcome=1 | absent) = (a*d)/(b*c)

    A Haldane-Anscombe correction (add 0.5 to every cell) is applied whenever
    any cell is zero, so the log-OR stays finite (with a wide CI).

    Returns dict: log_or, odds_ratio, se, ci_low, ci_high, corrected (bool).
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    corrected = min(a, b, c, d) == 0
    if corrected:
        a += correction; b += correction; c += correction; d += correction
    odds_ratio = (a * d) / (b * c)
    log_or = np.log(odds_ratio)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)        # SE of the log-OR
    ci_low = np.exp(log_or - 1.96 * se)
    ci_high = np.exp(log_or + 1.96 * se)
    return {
        "log_or": log_or,
        "odds_ratio": odds_ratio,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "corrected": corrected,
    }


def motif_contingency(present_mask, outcome):
    """
    Build the (a,b,c,d) contingency counts from a boolean presence mask and a
    binary outcome array, then return the log-odds-ratio dict augmented with
    the raw counts and the present/absent outcome rates (percentages).
    """
    present = np.asarray(present_mask).astype(bool)
    y = np.asarray(outcome).astype(int)
    a = int(np.sum(present & (y == 1)))
    b = int(np.sum(present & (y == 0)))
    c = int(np.sum(~present & (y == 1)))
    d = int(np.sum(~present & (y == 0)))
    res = log_odds_ratio(a, b, c, d)
    res.update({
        "a_present_pos": a, "b_present_neg": b,
        "c_absent_pos": c, "d_absent_neg": d,
        "pct_pos_when_present": 100 * a / (a + b) if (a + b) else np.nan,
        "pct_pos_when_absent":  100 * c / (c + d) if (c + d) else np.nan,
        "n_present": a + b, "n_absent": c + d,
    })
    return res
