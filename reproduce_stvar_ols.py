"""
reproduce_stvar_ols.py
==============================================================================
Which topology components drive the WITHIN-NETWORK VARIABILITY of settling time?

Target: rCV = IQR / median of stability_time across a network's HR parameter sets
at sigma 1.0, from nz_stvar_per_network<suffix>.csv (networks with >= 20 sets, the
MIN_RUNS rule of reproduce_noise). rCV is scale-free, so it is comparable across
categories whose settling times differ by an order of magnitude.

This is the variability analogue of the per-loop OLS that dyn_stability_ols runs on
the settling-time LEVEL. Same estimator as everywhere else in the project: OLS on
the six elementary feedback-loop counts with HC3 (heteroskedasticity-robust) errors,
reported as the change in rCV per additional loop.

Two nested models, both TOTAL effects:
  crude              ~ six loop counts
  density_adjusted   + n_edges          -- denser circuits carry more loops, so
                                          without this the loop terms partly measure
                                          density. Here density turns out to matter
                                          very little (n_edges n.s., R2 unchanged),
                                          which is unusual in this project.

A third model adding the settling-time LEVEL (st_median) was fitted and deliberately
DROPPED. It conditions on a mediator: level is itself downstream of topology
(dyn_stability_ols), so its coefficients are direct effects with the indirect path
removed, not total effects -- and rCV carries the median in its denominator, making
part of any level coefficient mechanical. What it showed, for the record: four of the
six loop terms fall to zero and R2 rises 0.36 -> 0.78, i.e. the loop-variability
relationship runs almost entirely THROUGH the level. That is a mediation result, not
a correction to the numbers reported here.

Single noise level by design: rCV falls steeply with noise and the MIN_RUNS filter
keeps very different network sets at each level (2,175 / 2,449 / 1,615 at sigma
0.2 / 1 / 5), so pooling or panelling levels would compare different populations.

Output: results/reproduce/dyn_stvar_ols<suffix>.csv
        results/reproduce/dyn_stvar_perloop_bars<suffix>.{png,svg}
==============================================================================
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import statsmodels.api as sm

from repro_common import (build_feature_table, savefig, ELEMENTARY, PRETTY, OUT_SUFFIX,
                          NZ_SUFFIX, RESULTS_FILE)

warnings.filterwarnings("ignore")
RES = "results/reproduce"
os.makedirs(RES, exist_ok=True)

LEVEL = 1.0
LAB, TICK, LEG = 10, 9, 8.5
POS, NEG = "#C62828", "#1565C0"
# barh index 0 = bottom, so list bottom-to-top: PFBL 1->2->3 on top, NFBL below
ORDER = ["fb_neg_3node", "fb_neg_2node", "fb_neg_1node",
         "fb_pos_3node", "fb_pos_2node", "fb_pos_1node"]
MODELS = ["crude", "density_adjusted"]
M_LABEL = {"crude": "Crude", "density_adjusted": "Density-adjusted"}
M_ALPHA = {"crude": 0.42, "density_adjusted": 1.0}

# --- data --------------------------------------------------------------------
sv = pd.read_csv(f"{RES}/nz_stvar_per_network{NZ_SUFFIX}.csv")
sv = sv[sv.noise == LEVEL][["net_id", "fb_group5", "n", "rcv"]]

feat = build_feature_table(results_file=RESULTS_FILE)
model_df = sv.merge(feat[["net_id", "n_edges"] + ELEMENTARY], on="net_id")
model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["rcv", "n_edges"] + ELEMENTARY)

print(f"rCV of settling time at sigma {LEVEL:g}: {len(model_df):,} networks "
      f"(>= 20 sets), median rCV {model_df.rcv.median():.3f}")


def fit_ols(cols, label):
    sub = model_df[["rcv"] + cols].dropna()
    fit = sm.OLS(sub["rcv"].astype(float).values,
                 sm.add_constant(sub[cols].astype(float))).fit(cov_type="HC3")
    ci = fit.conf_int()
    return pd.DataFrame({"feature": ["const"] + list(cols), "model": label,
                         "coef": fit.params.values,
                         "ci_low": ci[0].values, "ci_high": ci[1].values,
                         "pval": fit.pvalues.values,
                         "n": len(sub), "r2": fit.rsquared,
                         "r2_adj": fit.rsquared_adj})


tbl = pd.concat([fit_ols(ELEMENTARY, "crude"),
                 fit_ols(ELEMENTARY + ["n_edges"], "density_adjusted")],
                ignore_index=True)
tbl = tbl[tbl.feature != "const"].reset_index(drop=True)
tbl["pretty"] = tbl.feature.map(PRETTY).fillna(tbl.feature)
tbl.to_csv(f"{RES}/dyn_stvar_ols{OUT_SUFFIX}.csv", index=False)

for m in MODELS:
    s = tbl[tbl.model == m]
    print(f"\n[{M_LABEL[m]}]  R2 = {s.r2.iloc[0]:.3f} (adj {s.r2_adj.iloc[0]:.3f}), "
          f"n = {s.n.iloc[0]:,}")
    for _, r in s.iterrows():
        print(f"  {r.pretty:22s}: {r.coef:+8.4f} rCV per loop  "
              f"[{r.ci_low:+.4f},{r.ci_high:+.4f}]  (p={r.pval:.1e})")

# --- figure ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.9, 4.4))
h = 0.34
for j, m in enumerate(MODELS):
    d = tbl[tbl.model == m].set_index("feature")
    off = (j - 0.5) * h
    for i, f in enumerate(ORDER):
        r = d.loc[f]
        ax.barh(i + off, r.coef, height=h, color=POS if "pos" in f else NEG,
                alpha=M_ALPHA[m], edgecolor="white", linewidth=0.5,
                xerr=[[r.coef - r.ci_low], [r.ci_high - r.coef]],
                error_kw=dict(ecolor="#2b2b2b", lw=0.9, capsize=2.2), zorder=3)
ax.axvline(0, color="black", lw=1.0)
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels([PRETTY[f].replace(" (self)", "\n(self)") for f in ORDER], fontsize=TICK)
ax.set_xlabel("Change in rCV of Settling Time per Loop", fontsize=LAB)
ax.tick_params(axis="x", labelsize=TICK)
ax.locator_params(axis="x", nbins=5)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
ax.grid(axis="x", color="#EEEEEE", lw=0.8)

# two keys: hue = loop polarity (house convention), saturation = adjustment model
leg1 = ax.legend(handles=[mpatches.Patch(color=POS, label="Positive Feedback"),
                          mpatches.Patch(color=NEG, label="Negative Feedback")],
                 fontsize=LEG - 0.5, loc="lower right", frameon=False,
                 handlelength=1.2, handletextpad=0.5, borderaxespad=0.2)
ax.add_artist(leg1)
ax.legend(handles=[mpatches.Patch(facecolor="#777777", alpha=M_ALPHA[m], label=M_LABEL[m])
                   for m in MODELS],
          fontsize=LEG - 0.5, loc="upper left", frameon=False, handlelength=1.2,
          handletextpad=0.5, labelspacing=0.25, borderaxespad=0.2)

fig.tight_layout()
savefig(f"{RES}/dyn_stvar_perloop_bars{OUT_SUFFIX}", bbox_inches="tight")
print(f"\nSaved -> dyn_stvar_ols{OUT_SUFFIX}.csv + "
      f"dyn_stvar_perloop_bars{OUT_SUFFIX}.{{png,svg}}")
