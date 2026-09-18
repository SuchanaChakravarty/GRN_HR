"""
reproduce_hr_robustness_composition.py
==============================================================================
Robustness companion to reproduce_hr_composition (which did HR EXISTENCE). Outcome
here is HR ROBUSTNESS = per-parameter-set multimodal rate (multimodal_count / 1000).
Question: does polarity contribute to HR robustness differently across the 5-way
feedback-composition axis, once density is held fixed?

Answer: YES (unlike existence). The density-adjusted robustness is an ASYMMETRIC hump
peaking at Mixed Pos-Dom: at fixed density, positive-dominant mixed feedback is
significantly more robust than negative-dominant (mirror contrast Mixed Neg−Pos-Dom
≈ −0.26 log-OR, p < 0.001), while the pure categories are polarity-symmetric (n.s.).
So HR *existence* is governed by feedback quantity (polarity-symmetric), but HR
*robustness* carries an extra polarity tilt favouring positive dominance — consistent
with the per-loop robustness GLM (PFL > NFL per loop) and the noise thread.

Model: binomial GLM  count ~ Binomial(1000, p),  logit(p) = n_edges + n_edges^2 +
       C(fb_group5)  (Pure Positive = ref).
  - Density-adjusted rate per category = g-computation (set whole population to that
    category, predict per-set p, average over the real n_edges distribution).
  - Network-level bootstrap (resample networks) for CIs + mirror-contrast p — honours
    the severe network-level overdispersion (naive binomial SE would be far too small).

Built from the two raw inputs (feedback composition via feedback_counts + the corrected
baseline summary_result_noise1.csv, read chunked). Outputs:
    results/reproduce/hr_robustness_composition.csv
    results/reproduce/hr_robustness_composition.{png,svg}
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

from repro_common import (load_adjacency_matrices, feedback_group5, savefig, GROUP5,
                          edge_bearing_components, CONNECTED, OUT_SUFFIX, RESULTS_FILE)
from motif_lib import feedback_counts

warnings.filterwarnings("ignore")
os.makedirs("results/reproduce", exist_ok=True)
RES = "results/reproduce"
LAB, TICK, LEG = 13, 11, 10
N_TRIALS, NBOOT = 1000, 500
RNG = np.random.default_rng(0)
G5_COLOR = {"Pure Positive": "#B71C1C", "Mixed Pos-Dom": "#E57373", "Balanced": "#9E9E9E",
            "Mixed Neg-Dom": "#64B5F6", "Pure Negative": "#0D47A1"}
# compact labels: the full names collide at this width, and shortening them is
# preferable to shrinking the tick font
G5_LABEL = {"Pure Positive": "Pure\nPos", "Mixed Pos-Dom": "Mixed\nPos",
            "Balanced": "Balanced", "Mixed Neg-Dom": "Mixed\nNeg",
            "Pure Negative": "Pure\nNeg"}

# --- per-network multimodal count + composition -----------------------------
counts = {}
for ch in pd.read_csv(RESULTS_FILE, usecols=["net_id"], chunksize=100000):
    for nid, c in ch["net_id"].value_counts().items():
        counts[nid] = counts.get(nid, 0) + int(c)
rows = []
for uid, m in load_adjacency_matrices().items():
    fb = feedback_counts(m)
    rows.append({"n_edges": int(np.sum(m != 0)), "mmc": int(counts.get(uid, 0)),
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
df["rate"] = df.mmc / N_TRIALS

DUM = GROUP5[1:]                                    # ref = Pure Positive
X = pd.DataFrame({"n_edges": df.n_edges.astype(float), "n_edges2": (df.n_edges ** 2).astype(float)})
for g in DUM:
    X[g] = (df.fb_group5 == g).astype(float)
Xc = sm.add_constant(X)
endog = np.column_stack([df.mmc.values, N_TRIALS - df.mmc.values]).astype(float)


def fit_stats(Xc_, endog_):
    """Density-adjusted per-category rate (g-computation) + category log-OR coefs."""
    f = sm.GLM(endog_, Xc_, family=sm.families.Binomial()).fit()
    rates = {}
    for g in GROUP5:
        Z = Xc_.copy()
        for gg in DUM:
            Z[gg] = 1.0 if gg == g else 0.0
        rates[g] = float(f.predict(Z).mean())
    coef = {g: (f.params[g] if g in DUM else 0.0) for g in GROUP5}
    return rates, coef


adj, coef = fit_stats(Xc, endog)
raw = {g: df.loc[df.fb_group5 == g, "rate"].mean() for g in GROUP5}
ngrp = {g: int((df.fb_group5 == g).sum()) for g in GROUP5}
c_pp0 = coef["Pure Negative"] - coef["Pure Positive"]
c_mm0 = coef["Mixed Neg-Dom"] - coef["Mixed Pos-Dom"]

bootR = {g: [] for g in GROUP5}
bPP, bMM = [], []
for _ in range(NBOOT):
    idx = RNG.integers(0, len(df), len(df))
    r, c = fit_stats(Xc.iloc[idx].reset_index(drop=True), endog[idx])
    for g in GROUP5:
        bootR[g].append(r[g])
    bPP.append(c["Pure Negative"] - c["Pure Positive"])
    bMM.append(c["Mixed Neg-Dom"] - c["Mixed Pos-Dom"])
adjCI = {g: np.percentile(bootR[g], [2.5, 97.5]) for g in GROUP5}


def bp(vals):
    v = np.asarray(vals); f = np.mean(v > 0)
    return 2 * min(f, 1 - f)

pPP, pMM = bp(bPP), bp(bMM)
ciPP, ciMM = np.percentile(bPP, [2.5, 97.5]), np.percentile(bMM, [2.5, 97.5])

pd.DataFrame([{"fb_group5": g, "n": ngrp[g], "raw_rate": raw[g], "adj_rate": adj[g],
               "adj_ci_low": adjCI[g][0], "adj_ci_high": adjCI[g][1]} for g in GROUP5]
             ).to_csv(f"{RES}/hr_robustness_composition{OUT_SUFFIX}.csv", index=False)

print("Density-adjusted (g-computation) HR-ROBUSTNESS rate per category:")
for g in GROUP5:
    print(f"  {g:16s} (n={ngrp[g]:4d}): raw={raw[g]:.3f}  adjusted={adj[g]:.3f}  [{adjCI[g][0]:.3f},{adjCI[g][1]:.3f}]")
print("\nDensity-adjusted mirror contrasts (log-OR per set, network-bootstrap):")
print(f"  Pure Negative - Pure Positive : {c_pp0:+.3f}  95%CI [{ciPP[0]:+.3f},{ciPP[1]:+.3f}]  p={pPP:.3f}")
print(f"  Mixed Neg-Dom - Mixed Pos-Dom : {c_mm0:+.3f}  95%CI [{ciMM[0]:+.3f},{ciMM[1]:+.3f}]  p={pMM:.3f}")

# --- figure ------------------------------------------------------------------
x = np.arange(len(GROUP5))
fig, ax = plt.subplots(figsize=(3.96, 3.53))   # width -20% again; fonts unchanged
ax.plot(x, [raw[g] for g in GROUP5], "-", color="#CFCFCF", lw=1.4, zorder=1)
ax.plot(x, [adj[g] for g in GROUP5], "-", color="#999999", lw=1.6, zorder=1)
for i, g in enumerate(GROUP5):
    ax.plot(x[i], raw[g], "o", ms=9, mfc="white", mec=G5_COLOR[g], mew=1.8, zorder=2)
    ax.errorbar(x[i], adj[g], yerr=[[adj[g] - adjCI[g][0]], [adjCI[g][1] - adj[g]]], fmt="o",
                ms=11, color=G5_COLOR[g], ecolor="#555", elinewidth=1.4, capsize=4,
                capthick=1.4, mec="white", mew=1.2, zorder=3)
ax.set_ylabel("HR Robustness", fontsize=LAB)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax.set_xticks(x); ax.set_xticklabels([G5_LABEL[g] for g in GROUP5], fontsize=TICK)
ax.tick_params(axis="y", labelsize=TICK)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True); ax.grid(axis="y", color="#EEEEEE", lw=0.8)
# Contrasts go in the caption, not on the axes (still printed and in the CSV).
# Legend lower-centre: the hump peaks at Mixed Pos-Dom and both pure ends are low,
# leaving the bottom-middle band empty.
ax.legend(handles=[Line2D([0], [0], marker="o", color="#888", mfc="white", mec="#555", mew=1.8,
                          lw=1.4, label="Raw rate"),
                   Line2D([0], [0], marker="o", color="#888", mfc="#555", mec="white",
                          lw=1.6, label="Density-adjusted (95% CI)")],
          fontsize=LEG, loc="lower center", frameon=False, ncol=1)
fig.tight_layout()
savefig(f"{RES}/hr_robustness_composition{OUT_SUFFIX}", bbox_inches="tight")
print(f"\n[pool: {'connected-only' if CONNECTED else 'full'}]  Saved -> "
      f"hr_robustness_composition{OUT_SUFFIX}.csv + hr_robustness_composition{OUT_SUFFIX}.{{png,svg}}")
