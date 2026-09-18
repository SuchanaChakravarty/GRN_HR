"""
visualize_parsimony_prediction.py
==============================================================================
Presentation figure for the parsimony-prediction result, built from the CSVs
written by reproduce_parsimony_prediction.py (results/reproduce/). Pure
visualization -- no model fitting -- so it is fast and can be re-styled freely
without touching the (compute-heavy) reproduction.

Reads:
    results/reproduce/parsimony_presentation.csv       (robustness: R^2, Spearman)
    results/reproduce/parsimony_clf_presentation.csv    (existence: ROC-AUC, PR-AUC(min))
    results/reproduce/parsimony_reference.csv           (chance/baseline per metric)

Writes (png + svg, project convention):
    results/reproduce/parsimony_prediction_overview.{png,svg}

Design: two stacked clustered-bar panels sharing one x-axis of feature sets
(Shuffled -> Density -> Feedback -> Density + Feedback -> Full Topology). Top
panel = Existence, bottom = Robustness. Within each panel the clustered bars are
the two metrics for that question, coloured BY METRIC. The empirical "Shuffled"
cluster shows the null level for each metric; error bars are +/- SD across the 15
repeated-CV partitions. Text sizes and panel heights match feedback_roles_barchart.
==============================================================================
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repro_common import savefig, OUT_SUFFIX

RES = "results/reproduce"

# --- load (keep_default_na=False so text set names stay strings) --------------
reg = pd.read_csv(f"{RES}/parsimony_presentation{OUT_SUFFIX}.csv", keep_default_na=False).set_index("set")
clf = pd.read_csv(f"{RES}/parsimony_clf_presentation{OUT_SUFFIX}.csv", keep_default_na=False).set_index("set")

# --- shared layout ------------------------------------------------------------
ORDER   = ["shuffled", "density", "feedback", "density+feedback", "full"]
XLABELS = ["Shuffled", "Density", "Feedback", "Density +\nFeedback", "Full\nTopology"]
W = 0.38                                        # width of each clustered bar
TICK_FS, XTICK_FS, YLAB_FS, LEG_FS = 13, 13.5, 14, 12   # match feedback_roles_barchart

# metrics per panel: (legend label, column, colour) -- colour encodes METRIC TYPE
EXIST = [("ROC-AUC", "roc_auc", "#4C72B0"),
         ("Minority PR-AUC", "pr_auc_minority", "#DD8452")]
ROBUST = [("R²", "cv_r2", "#55A868"),
          ("Spearman", "spearman", "#C44E52")]


def clustered(ax, tbl, metrics, ylim, ylabel, yticks, zero_line, show_xlabels=True):
    xg = np.arange(len(ORDER))
    nb = len(metrics)
    for k, (lab, col, color) in enumerate(metrics):
        off = (k - (nb - 1) / 2) * W
        vals = [tbl.loc[s, col] for s in ORDER]
        errs = [tbl.loc[s, col + "_sd"] for s in ORDER]
        ax.bar(xg + off, vals, W, yerr=errs, color=color, edgecolor="white",
               error_kw=dict(ecolor="#2b2b2b", lw=1.3, capsize=3.5), label=lab, zorder=3)
    if zero_line:
        ax.axhline(0, color="black", lw=1.0, zorder=1)
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xticks(xg)
    ax.set_xticklabels(XLABELS if show_xlabels else [], fontsize=XTICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_ylabel(ylabel, fontsize=YLAB_FS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#EEEEEE", lw=0.8)
    ax.legend(fontsize=LEG_FS, loc="upper left", frameon=False,
              handletextpad=0.6, borderaxespad=0.4)


fig, (ax_e, ax_r) = plt.subplots(2, 1, figsize=(7.2, 8.6))

clustered(ax_e, clf, EXIST, ylim=(0, 1.05), ylabel="Existence",
          yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], zero_line=False, show_xlabels=False)
clustered(ax_r, reg, ROBUST, ylim=(-0.28, 1.05), ylabel="Robustness",
          yticks=[-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0], zero_line=True)

fig.tight_layout(h_pad=0.6)
savefig(f"{RES}/parsimony_prediction_overview{OUT_SUFFIX}", bbox_inches="tight")
print(f"Saved -> {RES}/parsimony_prediction_overview{OUT_SUFFIX}.{{png,svg}}")
