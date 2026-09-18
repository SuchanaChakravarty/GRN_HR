"""
reproduce_hr_composition.py
==============================================================================
HR EXISTENCE across the 5-way feedback-composition axis (Pure Positive -> Mixed
Pos-Dom -> Balanced -> Mixed Neg-Dom -> Pure Negative), RAW vs DENSITY-ADJUSTED.
Motivated by the pos/neg asymmetry in the density-controlled MH table
(reproduce_mh_table_standalone): is there a monotonic or mirror-asymmetric change
in support for HR existence across feedback composition, once density is held fixed?

Answer: NO. The raw per-category HR-existence rate is a symmetric M-shape driven by
loop count / density (pure = few loops = low; mixed = many loops = high). A
density-adjusted logistic (label ~ n_edges + n_edges^2 + category) collapses the
24-point raw spread to a ~6-point band, and the two mirror contrasts
(Pure Neg-Pos, Mixed Neg-Pos) are both small, positive (direction consistent with
the MH NFL>PFL asymmetry) but non-significant. HR existence is governed by feedback
QUANTITY (density), not polarity order.

  - Density-adjusted rate per category = g-computation (set the whole population to
    that category, predict, average over the real n_edges distribution); bootstrap CI.
  - Mirror contrasts on the density-adjusted log-OR scale (Wald from the fit).

Built from the two raw inputs (feedback composition via feedback_counts + the
corrected baseline summary_result_noise1.csv, read chunked). Outputs:
    results/reproduce/hr_composition.csv
    results/reproduce/hr_composition.{png,svg}
==============================================================================
"""
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from repro_common import (load_adjacency_matrices, feedback_group5, savefig, GROUP5,
                          edge_bearing_components, CONNECTED, OUT_SUFFIX, RESULTS_FILE)
from motif_lib import feedback_counts

warnings.filterwarnings("ignore")
os.makedirs("results/reproduce", exist_ok=True)
RES = "results/reproduce"
LAB, TICK, LEG = 13, 11, 10
NBOOT = 500
RNG = np.random.default_rng(0)
G5_COLOR = {"Pure Positive": "#B71C1C", "Mixed Pos-Dom": "#E57373", "Balanced": "#9E9E9E",
            "Mixed Neg-Dom": "#64B5F6", "Pure Negative": "#0D47A1"}
# compact labels: the full names collide at this width, and shortening them is
# preferable to shrinking the tick font
G5_LABEL = {"Pure Positive": "Pure\nPos", "Mixed Pos-Dom": "Mixed\nPos",
            "Balanced": "Balanced", "Mixed Neg-Dom": "Mixed\nNeg",
            "Pure Negative": "Pure\nNeg"}

# --- per-network label + composition ----------------------------------------
counts = {}
for ch in pd.read_csv(RESULTS_FILE, usecols=["net_id"], chunksize=100000):
    for nid, c in ch["net_id"].value_counts().items():
        counts[nid] = counts.get(nid, 0) + int(c)
rows = []
for uid, m in load_adjacency_matrices().items():
    fb = feedback_counts(m)
    rows.append({"n_edges": int(np.sum(m != 0)), "label": int(counts.get(uid, 0) > 0),
                 "disjoint": int(edge_bearing_components(m) >= 2),
                 "has_fb_pos_feedback": fb["has_pos_feedback"],
                 "has_fb_neg_feedback": fb["has_neg_feedback"],
                 "fb_pos_feedback": fb["pos_feedback"], "fb_neg_feedback": fb["neg_feedback"]})
df = pd.DataFrame(rows)
if CONNECTED:
    df = df[df.disjoint == 0]                        # drop genuinely-disjoint networks
p, ng = df.has_fb_pos_feedback == 1, df.has_fb_neg_feedback == 1
df["category"] = np.select([p & ~ng, ng & ~p, p & ng], ["PFL-only", "NFL-only", "Mixed"],
                           default="No feedback")
df["fb_group5"] = feedback_group5(df)
df = df[df.fb_group5.isin(GROUP5)].reset_index(drop=True)

# design: smooth density (n_edges + n_edges^2) + category dummies (ref = Pure Positive)
DUM = GROUP5[1:]
X = pd.DataFrame({"n_edges": df.n_edges.astype(float), "n_edges2": (df.n_edges ** 2).astype(float)})
for g in DUM:
    X[g] = (df.fb_group5 == g).astype(float)
Xc = sm.add_constant(X)
y = df.label.values.astype(float)


def adjusted_rates(Xc_, y_):
    """g-computation density-adjusted HR-existence rate per category (+ the fit)."""
    fit = sm.Logit(y_, Xc_).fit(disp=0)
    out = {}
    for g in GROUP5:
        Z = Xc_.copy()
        for gg in DUM:
            Z[gg] = 1.0 if gg == g else 0.0
        out[g] = float(fit.predict(Z).mean())
    return out, fit


adj, fit = adjusted_rates(Xc, y)
raw = {g: df.loc[df.fb_group5 == g, "label"].mean() for g in GROUP5}
ngrp = {g: int((df.fb_group5 == g).sum()) for g in GROUP5}

boot = {g: [] for g in GROUP5}
for _ in range(NBOOT):
    idx = RNG.integers(0, len(df), len(df))
    a, _ = adjusted_rates(Xc.iloc[idx].reset_index(drop=True), y[idx])
    for g in GROUP5:
        boot[g].append(a[g])
adjCI = {g: np.percentile(boot[g], [2.5, 97.5]) for g in GROUP5}

cov = fit.cov_params()

def contrast(a, b):     # coef(a) - coef(b) density-adjusted log-OR (Pure Positive coef = 0)
    ca = fit.params[a] if a in DUM else 0.0
    cb = fit.params[b] if b in DUM else 0.0
    va = cov.loc[a, a] if a in DUM else 0.0
    vb = cov.loc[b, b] if b in DUM else 0.0
    cab = cov.loc[a, b] if (a in DUM and b in DUM) else 0.0
    d = ca - cb
    se = np.sqrt(max(va + vb - 2 * cab, 0.0))
    z = d / se if se > 0 else np.nan
    return d, se, 2 * stats.norm.sf(abs(z))

# --- persist the table -------------------------------------------------------
pd.DataFrame([{"fb_group5": g, "n": ngrp[g], "raw_rate": raw[g], "adj_rate": adj[g],
               "adj_ci_low": adjCI[g][0], "adj_ci_high": adjCI[g][1]} for g in GROUP5]
             ).to_csv(f"{RES}/hr_composition{OUT_SUFFIX}.csv", index=False)

print("Density-adjusted (g-computation) HR-existence rate per category:")
for g in GROUP5:
    print(f"  {g:16s} (n={ngrp[g]:4d}): raw={raw[g]:.3f}  adjusted={adj[g]:.3f}"
          f"  [{adjCI[g][0]:.3f},{adjCI[g][1]:.3f}]")
print("\nDensity-adjusted mirror contrasts (log-OR, Wald):")
for a, b in [("Pure Negative", "Pure Positive"), ("Mixed Neg-Dom", "Mixed Pos-Dom")]:
    d, se, pv = contrast(a, b)
    print(f"  {a:14s} - {b:14s}: dlog-OR={d:+.3f}  (SE {se:.3f}, p={pv:.2e})")

# --- figure: raw vs density-adjusted per-category HR existence ----------------
x = np.arange(len(GROUP5))
fig, ax = plt.subplots(figsize=(3.96, 3.53))   # width -20% again; fonts unchanged
ax.plot(x, [raw[g] for g in GROUP5], "-", color="#CFCFCF", lw=1.4, zorder=1)
ax.plot(x, [adj[g] for g in GROUP5], "-", color="#999999", lw=1.6, zorder=1)
for i, g in enumerate(GROUP5):
    ax.plot(x[i], raw[g], "o", ms=9, mfc="white", mec=G5_COLOR[g], mew=1.8, zorder=2)   # raw = open
    ax.errorbar(x[i], adj[g], yerr=[[adj[g] - adjCI[g][0]], [adjCI[g][1] - adj[g]]], fmt="o",
                ms=11, color=G5_COLOR[g], ecolor="#555", elinewidth=1.4, capsize=4,
                capthick=1.4, mec="white", mew=1.2, zorder=3)                            # adjusted = filled
ax.set_ylabel("HR Existence", fontsize=LAB)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax.set_xticks(x); ax.set_xticklabels([G5_LABEL[g] for g in GROUP5], fontsize=TICK)
ax.tick_params(axis="y", labelsize=TICK)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True); ax.grid(axis="y", color="#EEEEEE", lw=0.8)

# The mirror contrasts are printed to stdout and written to the CSV; they belong in
# the caption rather than on the axes. Legend sits lower-centre: the M-shape leaves
# that corner empty (the two pure categories are the low points, at the far ends).
ax.legend(handles=[Line2D([0], [0], marker="o", color="#888", mfc="white", mec="#555", mew=1.8,
                          lw=1.4, label="Raw rate"),
                   Line2D([0], [0], marker="o", color="#888", mfc="#555", mec="white",
                          lw=1.6, label="Density-adjusted (95% CI)")],
          fontsize=LEG, loc="lower center", frameon=False, ncol=1)
fig.tight_layout()
savefig(f"{RES}/hr_composition{OUT_SUFFIX}", bbox_inches="tight")
print(f"\n[pool: {'connected-only' if CONNECTED else 'full'}]  "
      f"Saved -> hr_composition{OUT_SUFFIX}.csv + hr_composition{OUT_SUFFIX}.{{png,svg}}")
