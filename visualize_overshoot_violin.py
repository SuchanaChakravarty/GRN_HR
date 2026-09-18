"""
visualize_overshoot_violin.py
==============================================================================
The OVERSHOOT panel of dyn_dist_violin, on its own.

dyn_dist_violin_overshoot is a stacked pair (settling time on top, overshoot below)
at 6.5 x 9.0 in. This draws only the overshoot panel, at 6.5 x 4.5 in -- one panel
of that figure, including the h_pad, so the axes box matches what the paired figure
produces. Style, colours, fonts, box-in-violin, the adjacent-neighbour significance
brackets and the trend annotation are all copied from violin_panel in
visualize_stability_dynamics so the two are interchangeable.

Outcome: per-network OVERSHOOT FRACTION -- the share of a network's HR parameter sets
whose cell-fraction trajectory peaks above its final value by the detector threshold.
Drawn on a PERCENT axis, since the request was for percentages; the underlying column
(over_rate) is the same 0-1 fraction the rest of the overshoot thread uses.

Inputs (both already built on the frac10 + op_ic criteria, nothing is recomputed):
    dyn_per_network_overshoot<suffix>.csv      per-network over_rate
    dyn_gradient_stats_overshoot<suffix>.csv   trend rho + adjacent Holm-corrected MWU

Output: results/reproduce/dyn_overshoot_violin<suffix>.{png,svg}
==============================================================================
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from repro_common import savefig, OUT_SUFFIX

warnings.filterwarnings("ignore")
RES = "results/reproduce"
os.makedirs(RES, exist_ok=True)

LAB, TICK = 15, 13                   # same enlarged fonts as visualize_stability_dynamics
GROUP5 = ["Pure Positive", "Mixed Pos-Dom", "Balanced", "Mixed Neg-Dom", "Pure Negative"]
G5_LABEL = ["Pure\nPositive", "Mixed\nPos-Dom", "Balanced", "Mixed\nNeg-Dom", "Pure\nNegative"]
G5_COLOR = {"Pure Positive": "#B71C1C", "Mixed Pos-Dom": "#E57373", "Balanced": "#9E9E9E",
            "Mixed Neg-Dom": "#64B5F6", "Pure Negative": "#0D47A1"}
YCOL, METRIC = "over_rate", "overshoot"


def signed(x, d=2):                  # true minus sign (U+2212), not a hyphen
    return f"{x:+.{d}f}".replace("-", "−")


def _stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


net = pd.read_csv(f"{RES}/dyn_per_network_{METRIC}{OUT_SUFFIX}.csv", keep_default_na=False)
net[YCOL] = pd.to_numeric(net[YCOL], errors="coerce")
grad = pd.read_csv(f"{RES}/dyn_gradient_stats_{METRIC}{OUT_SUFFIX}.csv")
g = grad[grad.metric == METRIC]
rho = g.loc[g.comparison == "trend(all groups)", "spearman_rho"].iloc[0]
star_list = [_stars(g.loc[g.comparison == f"{GROUP5[i]} -> {GROUP5[i+1]}",
                          "mwu_p_holm"].iloc[0]) for i in range(4)]

data = [net.loc[net.fb_group5 == grp, YCOL].dropna().values for grp in GROUP5]
xpos = np.arange(5)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
vp = ax.violinplot(data, positions=xpos, widths=0.82, showextrema=False)
for body, grp in zip(vp["bodies"], GROUP5):
    body.set_facecolor(G5_COLOR[grp]); body.set_alpha(0.30)
    body.set_edgecolor(G5_COLOR[grp]); body.set_linewidth(1.2)
bp = ax.boxplot(data, positions=xpos, widths=0.13, patch_artist=True, showfliers=False,
                medianprops=dict(color="white", lw=1.6),
                boxprops=dict(edgecolor="#2b2b2b", lw=1.0),
                whiskerprops=dict(color="#2b2b2b", lw=1.0),
                capprops=dict(color="#2b2b2b", lw=1.0))
for patch, grp in zip(bp["boxes"], GROUP5):
    patch.set_facecolor(G5_COLOR[grp])

# top limit above the highest box-whisker (1.5*IQR cap) + headroom for brackets
ymin = 0.0
tops = []
for d in data:
    q1, q3 = np.percentile(d, [25, 75])
    tops.append(d[d <= q3 + 1.5 * (q3 - q1)].max())
data_top = max(tops); span = data_top - ymin
ax.set_ylim(ymin, data_top + span * 0.17)

br = data_top + span * 0.05          # adjacent-neighbour brackets (Holm-corrected MWU)
for i, s in enumerate(star_list):
    ax.plot([i + 0.12, i + 0.88], [br, br], color="#666", lw=1.1)
    ax.text(i + 0.5, br + span * (0.012 if s != "ns" else 0.02), s, ha="center",
            va="bottom", fontsize=13 if s != "ns" else 9.5, color="#333")
ax.text(0.98, 0.03, f"Trend: Spearman ρ = {signed(rho)}", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=11.5, style="italic", color="#333")

ax.set_xticks(xpos); ax.set_xticklabels(G5_LABEL, fontsize=TICK)
ax.set_ylabel("Overshoot Percentage of HR Sets", fontsize=LAB)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax.tick_params(axis="y", labelsize=TICK)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True); ax.grid(axis="y", color="#EEEEEE", lw=0.8)

fig.tight_layout()
savefig(f"{RES}/dyn_overshoot_violin{OUT_SUFFIX}", bbox_inches="tight")
for grp, d in zip(GROUP5, data):
    print(f"  {grp:16s} n={len(d):4d}  median {np.median(d):6.1%}  "
          f"IQR [{np.percentile(d, 25):.1%}, {np.percentile(d, 75):.1%}]")
print(f"\nTrend across ordered groups: Spearman rho = {rho:+.3f}")
print(f"Saved -> dyn_overshoot_violin{OUT_SUFFIX}.{{png,svg}}")
