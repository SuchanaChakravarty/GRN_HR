"""
visualize_stability_dynamics.py
==============================================================================
Figures for the Phases 8-10 dynamics reproduction (from results/reproduce/dyn_*),
with SETTLING TIME and PRE-SETTLING FLUCTUATION as matched analyses. Feedback is
grouped 5 ways (Pure Positive -> Mixed Pos-Dom -> Balanced -> Mixed Neg-Dom ->
Pure Negative). Pure plotting.

Four figures, each two panels top (settling time) / bottom (fluctuation):
  dyn_perloop_bars.{png,svg}     per-loop OLS effect of each feedback type (density-adjusted, 95% CI)
  dyn_rate_forest.{png,svg}      rate<->x Spearman stratified by category (95% CI)
  dyn_dist_violin.{png,svg}      per-network distribution by the 5 feedback groups (violin + inner box)
  dyn_rate_scatter.{png,svg}     rate vs x, coloured by feedback group, trend + 95% CI + per-group rho

Colour convention: positive feedback = reddish, negative feedback = blueish.
==============================================================================
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from repro_common import savefig, PRETTY, OUT_SUFFIX, METRIC2, M2_TAG, M2_COL

RES = "results/reproduce"
LAB, TICK, LEG = 15, 13, 12          # enlarged label / tick / legend fonts

def signed(x, d=2):                  # signed number with a true minus sign (U+2212), not a hyphen
    return f"{x:+.{d}f}".replace("-", "−")

# --- colour convention: positive -> reddish, negative -> blueish --------------
POS, NEG = "#C0625E", "#5E7FA6"          # match feedback_roles_barchart (PFBL red / NFBL blue)
GROUP5 = ["Pure Positive", "Mixed Pos-Dom", "Balanced", "Mixed Neg-Dom", "Pure Negative"]
G5_LABEL = ["Pure\nPositive", "Mixed\nPos-Dom", "Balanced", "Mixed\nNeg-Dom", "Pure\nNegative"]
G5_COLOR = {"Pure Positive": "#B71C1C", "Mixed Pos-Dom": "#E57373", "Balanced": "#9E9E9E",
            "Mixed Neg-Dom": "#64B5F6", "Pure Negative": "#0D47A1"}
G5_ALPHA = {"Pure Positive": 0.45, "Mixed Pos-Dom": 0.16, "Balanced": 0.30,
            "Mixed Neg-Dom": 0.16, "Pure Negative": 0.45}

# forest: colour by category SYSTEM (unrelated to polarity), accessible names + title case
SYS_COLOR = {"SCC profile": "#00897B", "Density": "#8E24AA", "Feedback": "#F9A825"}
SYS_DISPLAY = {"SCC profile": "Connectivity", "Density": "Density", "Feedback": "Feedback"}
ACC = {"(3) strongly conn.": "3-Gene Loop", "(2,1)": "2-Gene Loop", "(1,1,1)": "No Multi-Gene Loop",
       "<=4 edges": "≤4 Edges", "5 edges": "5 Edges", "6 edges": "6 Edges",
       ">=7 edges": "≥7 Edges"}
def acc(name):
    return ACC.get(name, name)

ELEM = ["fb_pos_1node", "fb_neg_1node", "fb_pos_2node", "fb_neg_2node",
        "fb_pos_3node", "fb_neg_3node"]

net = pd.read_csv(f"{RES}/dyn_per_network{M2_TAG}{OUT_SUFFIX}.csv", keep_default_na=False)
for c in ["st_median", M2_COL, "multimodal_rate", "within_corr"]:
    net[c] = pd.to_numeric(net[c], errors="coerce")
if "fb_group5" not in net.columns:
    net["fb_group5"] = np.select(
        [net.category == "PFL-only", net.category == "NFL-only",
         net.polarity_class == "pos-dominant", net.polarity_class == "balanced"],
        ["Pure Positive", "Pure Negative", "Mixed Pos-Dom", "Balanced"], default="Mixed Neg-Dom")


# =============================================================================
# Figure 1 — per-loop OLS bars (top: settling time, bottom: fluctuation)
# =============================================================================
st_ols = pd.read_csv(f"{RES}/dyn_stability_ols{OUT_SUFFIX}.csv")
fl_ols = pd.read_csv(f'{RES}/{METRIC2["ols"]}{OUT_SUFFIX}.csv')
# fixed order: PFBL (1->2->3-node) on top, NFBL (1->2->3-node) below; same in both panels.
# barh index 0 = bottom, so list bottom-to-top (reverse of the desired top-to-bottom order).
order = ["fb_neg_3node", "fb_neg_2node", "fb_neg_1node",
         "fb_pos_3node", "fb_pos_2node", "fb_pos_1node"]

def bar_panel(ax, tbl, xlabel):
    adj = tbl[tbl.model == "density_adjusted"].set_index("feature")
    for i, f in enumerate(order):
        r = adj.loc[f]
        ax.barh(i, r["coef"], color=POS if "pos" in f else NEG, edgecolor="white",
                xerr=[[r["coef"] - r["ci_low"]], [r["ci_high"] - r["coef"]]],
                error_kw=dict(ecolor="#2b2b2b", lw=1.2, capsize=3.5), zorder=3)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([PRETTY[f].replace(" (self)", "\n(self)") for f in order], fontsize=TICK)
    ax.set_xlabel(xlabel, fontsize=LAB)
    ax.tick_params(axis="x", labelsize=TICK)
    ax.locator_params(axis="x", nbins=6)          # thin x-ticks so labels don't crowd when narrow
    ax.spines[["top", "right"]].set_visible(False)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.3, 8.6))
bar_panel(ax1, st_ols, "Change in Median Settling Time per Loop (a.u.)")
bar_panel(ax2, fl_ols, METRIC2["perloop"])
ax1.legend(handles=[mpatches.Patch(color=POS, label="Positive Feedback"),
                    mpatches.Patch(color=NEG, label="Negative Feedback")],
           fontsize=LEG, loc="upper left", frameon=False)
fig.tight_layout(h_pad=2.2)
savefig(f"{RES}/dyn_perloop_bars{M2_TAG}{OUT_SUFFIX}", bbox_inches="tight")
print(f"Saved -> dyn_perloop_bars{M2_TAG}{OUT_SUFFIX}.{{png,svg}}")


# =============================================================================
# Figure 2 — stratified rate<->x forest (top: settling time, bottom: fluctuation)
# =============================================================================
def forest_panel(ax, csv, xlabel):
    t = pd.read_csv(csv)
    overall = t[t.system == "Overall"]["spearman"].iloc[0]
    strat = t[t.system != "Overall"].reset_index(drop=True)
    yb = np.arange(len(strat))[::-1]
    for y, (_, r) in zip(yb, strat.iterrows()):
        c = SYS_COLOR[r["system"]]
        ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=c, lw=2.2, zorder=2)
        ax.scatter(r["spearman"], y, color=c, s=68, edgecolor="white", zorder=3)
    ax.axvline(0, color="black", lw=1.0)
    ax.axvline(overall, color="#616161", ls="--", lw=1.3)
    ax.set_yticks(yb)
    ax.set_yticklabels([f"{acc(r['category'])}  (n={r['n']})" for _, r in strat.iterrows()],
                       fontsize=TICK - 1)
    ax.set_xlabel(xlabel, fontsize=LAB)
    ax.tick_params(axis="x", labelsize=TICK)
    ax.spines[["top", "right"]].set_visible(False)
    return overall

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 11.5))
ov1 = forest_panel(ax1, f"{RES}/dyn_rate_timing_strat{OUT_SUFFIX}.csv", "Spearman(Rate, Median Settling Time)")
ov2 = forest_panel(ax2, f'{RES}/{METRIC2["strat"]}{OUT_SUFFIX}.csv', METRIC2["forest"])
handles = [plt.Line2D([0], [0], color=c, lw=3, label=SYS_DISPLAY[s]) for s, c in SYS_COLOR.items()]
handles.append(plt.Line2D([0], [0], color="#616161", ls="--", lw=1.3, label="Overall"))
fig.tight_layout(h_pad=2.2, rect=[0, 0.035, 1, 1])
fig.legend(handles=handles, fontsize=LEG, loc="lower center", ncol=4,
           frameon=False, bbox_to_anchor=(0.5, 0.0))
savefig(f"{RES}/dyn_rate_forest{M2_TAG}{OUT_SUFFIX}", bbox_inches="tight")
print(f"Saved -> dyn_rate_forest{M2_TAG}{OUT_SUFFIX}.{{png,svg}}")


# =============================================================================
# Figure 3 — distribution by 5 feedback groups (top: settling time, bottom: fluctuation)
# =============================================================================
grad = pd.read_csv(f"{RES}/dyn_gradient_stats{M2_TAG}{OUT_SUFFIX}.csv")
def _stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
def gradient_annot(mlabel):
    g = grad[grad.metric == mlabel]
    rho = g.loc[g.comparison == "trend(all groups)", "spearman_rho"].iloc[0]
    st = [_stars(g.loc[g.comparison == f"{GROUP5[i]} -> {GROUP5[i+1]}", "mwu_p_holm"].iloc[0])
          for i in range(4)]
    return rho, st


def violin_panel(ax, ycol, ylabel, ymin, rho, star_list):
    data = [net.loc[net.fb_group5 == g, ycol].dropna().values for g in GROUP5]
    xpos = np.arange(5)
    vp = ax.violinplot(data, positions=xpos, widths=0.82, showextrema=False)
    for body, g in zip(vp["bodies"], GROUP5):
        body.set_facecolor(G5_COLOR[g]); body.set_alpha(0.30)
        body.set_edgecolor(G5_COLOR[g]); body.set_linewidth(1.2)
    bp = ax.boxplot(data, positions=xpos, widths=0.13, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", lw=1.6),
                    boxprops=dict(edgecolor="#2b2b2b", lw=1.0),
                    whiskerprops=dict(color="#2b2b2b", lw=1.0),
                    capprops=dict(color="#2b2b2b", lw=1.0))
    for patch, g in zip(bp["boxes"], GROUP5):
        patch.set_facecolor(G5_COLOR[g])
    # top limit above the highest box-whisker (1.5*IQR cap) + headroom for brackets
    tops = []
    for d in data:
        q1, q3 = np.percentile(d, [25, 75])
        tops.append(d[d <= q3 + 1.5 * (q3 - q1)].max())
    data_top = max(tops); span = data_top - ymin
    ax.set_ylim(ymin, data_top + span * 0.17)
    # adjacent-neighbor significance brackets (Holm-corrected Mann-Whitney)
    br = data_top + span * 0.05
    for i, s in enumerate(star_list):
        ax.plot([i + 0.12, i + 0.88], [br, br], color="#666", lw=1.1)
        ax.text(i + 0.5, br + span * (0.012 if s != "ns" else 0.02), s, ha="center",
                va="bottom", fontsize=13 if s != "ns" else 9.5, color="#333")
    # monotone-trend statistic across the ordered groups
    ax.text(0.98, 0.03, f"Trend: Spearman ρ = {signed(rho)}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11.5, style="italic", color="#333")
    ax.set_xticks(xpos); ax.set_xticklabels(G5_LABEL, fontsize=TICK)
    ax.set_ylabel(ylabel, fontsize=LAB)
    ax.tick_params(axis="y", labelsize=TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis="y", color="#EEEEEE", lw=0.8)

rho_s, star_s = gradient_annot("settling_time")
rho_f, star_f = gradient_annot(METRIC2["grad"])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 9.0))
# floor the settling-time axis at 500 rather than at the global minimum: a handful of
# very-fast networks would otherwise stretch the axis and leave the bulk (1200-1900)
# squeezed into the top third. Their violins run off the bottom edge rather than being
# dropped, and the KDEs are still fitted on the full data.
violin_panel(ax1, "st_median", "Settling Time (a.u.)", 500, rho_s, star_s)
violin_panel(ax2, M2_COL, METRIC2["axis"], 0, rho_f, star_f)
fig.tight_layout(h_pad=2.2)
savefig(f"{RES}/dyn_dist_violin{M2_TAG}{OUT_SUFFIX}", bbox_inches="tight")
print(f"Saved -> dyn_dist_violin{M2_TAG}{OUT_SUFFIX}.{{png,svg}}")


# =============================================================================
# Figure 4 — rate vs x scatter, 5 groups, trend + CI (top: settling, bottom: fluctuation)
# =============================================================================
def trend_ci(ax, x, y, color):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    fit = sm.OLS(y[m], sm.add_constant(x[m])).fit()
    xs = np.linspace(x[m].min(), x[m].max(), 100)
    pr = fit.get_prediction(sm.add_constant(xs)); ci = pr.conf_int()
    ax.fill_between(xs, ci[:, 0], ci[:, 1], color=color, alpha=0.20, lw=0, zorder=4)
    ax.plot(xs, pr.predicted_mean, color=color, lw=2.4, zorder=5)

DRAW = ["Mixed Pos-Dom", "Mixed Neg-Dom", "Balanced", "Pure Positive", "Pure Negative"]

def scatter_panel(ax, qcol, qlabel, legloc):
    for g in DRAW:
        sub = net[net.fb_group5 == g].dropna(subset=["multimodal_rate", qcol])
        ax.scatter(sub[qcol], sub.multimodal_rate, s=9, alpha=G5_ALPHA[g],
                   color=G5_COLOR[g], linewidths=0, zorder=2)
        trend_ci(ax, sub[qcol], sub.multimodal_rate, G5_COLOR[g])
    for g in GROUP5:
        sub = net[net.fb_group5 == g].dropna(subset=["multimodal_rate", qcol])
        rho = stats.spearmanr(sub[qcol], sub.multimodal_rate).correlation
        ax.plot([], [], color=G5_COLOR[g], lw=2.4, label=f"{g}  (ρ={signed(rho)})")
    ax.set_xlim(net[qcol].quantile(0.005), net[qcol].quantile(0.99))
    ax.set_ylim(0, net.multimodal_rate.quantile(0.99))
    ax.set_xlabel(qlabel, fontsize=LAB)
    ax.set_ylabel("Robustness", fontsize=LAB)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(labelsize=TICK)
    # translucent white backing so the legend stays readable over the denser clipped view
    ax.legend(fontsize=LEG, loc=legloc, frameon=True, facecolor="white",
              framealpha=0.8, edgecolor="none")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_box_aspect(1)                       # square-like axes

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 12.6))
scatter_panel(ax1, "st_median", "Median Settling Time (a.u.)", "upper right")
scatter_panel(ax2, M2_COL, METRIC2["axis_median"], "upper left")
fig.tight_layout(h_pad=2.0)
savefig(f"{RES}/dyn_rate_scatter{M2_TAG}{OUT_SUFFIX}", bbox_inches="tight")
print(f"Saved -> dyn_rate_scatter{M2_TAG}{OUT_SUFFIX}.{{png,svg}}")


# =============================================================================
# Figure 5 — within-network settling<->fluctuation Spearman, one histogram / group
# =============================================================================
# Each network (>=20 param sets) has a within-network Spearman(stability_time,
# cv_before_x1); overlay the 5 feedback groups' distributions (density-normalised
# so unequal group sizes don't dominate). Consistently positive across groups.
wc = net.dropna(subset=["within_corr"])
bins = np.linspace(wc.within_corr.min(), wc.within_corr.max(), 31)
fig, ax = plt.subplots(figsize=(6.6, 5.2))
for g in GROUP5:
    v = wc.loc[wc.fb_group5 == g, "within_corr"].values
    ax.hist(v, bins=bins, density=True, histtype="step", lw=2.2, color=G5_COLOR[g],
            label=f"{g}  (median {signed(np.median(v))})")
ax.axvline(0, color="black", lw=1.2, ls="--")
ax.set_xlabel(METRIC2["within"], fontsize=LAB)
ax.set_ylabel("Density", fontsize=LAB)
ax.tick_params(labelsize=TICK)
ax.legend(fontsize=LEG - 1, loc="upper left", frameon=True, facecolor="white",
          framealpha=0.85, edgecolor="none")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
savefig(f"{RES}/dyn_within_corr_hist{M2_TAG}{OUT_SUFFIX}", bbox_inches="tight")
print(f"Saved -> dyn_within_corr_hist{M2_TAG}{OUT_SUFFIX}.{{png,svg}}")
