"""
visualize_noise_traversal.py
==============================================================================
The noise thread for the PRE-SETTLING TRAVERSAL RATE, in the visual language of
nz_noise_panels: one-sided KDE ridges per feedback group at each noise level, a
MEDIAN trend threaded across levels, and marker area proportional to the number
of networks (or parameter sets) behind each median.

Two panels, matching the nz_noise_panels axes that have a traversal analogue:

  left    Traversal rate, per-network median      (mirrors the HR Settling Time
                                                   per-network-median panel)
  right   Traversal rate, all parameter sets      (mirrors the all-sets panel;
                                                   shares the left panel's y-axis)

The other two nz_noise_panels axes are dropped: HR robustness is a property of the
network rather than of its trajectories and has no traversal counterpart, and the
rCV panel is omitted by choice. The per-network rCV is still computed and written
to the CSV (column `rcv`, networks with >= MIN_RUNS sets).

Scale choices, checked against the data rather than copied:
  * The traversal rate is only mildly skewed -- it spans 1.27 decades from the
    0.2nd to the 99.8th percentile at sigma 0.2 and 0.52 decades at sigma 5 -- so
    both panels are LINEAR, unlike the log rCV axis of the original.
  * rCV requires MIN_RUNS sets per network, exactly as in reproduce_noise.

Sets settling in under MIN_SETTLING time units are dropped, as in path_per_network:
dividing by a near-zero settling time yields rates ~1000x the typical value.

Dynamics are read from the ORIGINAL-IC rows. The op_ic criterion selects WHICH
parameter sets count as HR; the trajectory columns in those rows are still the
original-IC simulation, which is what these metrics describe.

Output: results/reproduce/nz_traversal_panels<suffix>.{png,svg}
        results/reproduce/nz_traversal_per_network<suffix>.csv
==============================================================================
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from repro_common import (savefig, GROUP5, NOISE_LEVELS, NOISE_FILES, NZ_SUFFIX, NOISE_SET,
                          PATH_RATE_SCALE, load_adjacency_matrices, feedback_group5,
                          edge_bearing_components, METRIC2)
from motif_lib import feedback_counts

warnings.filterwarnings("ignore")
RES = "results/reproduce"
os.makedirs(RES, exist_ok=True)

LAB, TICK, LEG = 10, 9, 8.0
W = 0.80                       # one-sided KDE width (extends right)
KDE_CAP = 40000                # subsample cap for the set-level KDE
ALPHA = 0.20
SMAX = 170.0                   # marker area for the largest count
MIN_RUNS = 20                  # min sets per network for a variability estimate
MIN_SETTLING = 10.0
GENES = ["x1", "x2", "x3"]
LEVELS = NOISE_LEVELS
XPOS = range(len(LEVELS))
XLIM_RIDGE = (-0.35, len(LEVELS) - 1 + W + 0.12)
G5_COLOR = {"Pure Positive": "#B71C1C", "Mixed Pos-Dom": "#E57373", "Balanced": "#9E9E9E",
            "Mixed Neg-Dom": "#64B5F6", "Pure Negative": "#0D47A1"}

# --- topology ----------------------------------------------------------------
rows = []
for uid, m in load_adjacency_matrices().items():
    fb = feedback_counts(m)
    rows.append({"net_id": uid, "disjoint": int(edge_bearing_components(m) >= 2),
                 "has_fb_pos_feedback": fb["has_pos_feedback"],
                 "has_fb_neg_feedback": fb["has_neg_feedback"],
                 "fb_pos_feedback": fb["pos_feedback"],
                 "fb_neg_feedback": fb["neg_feedback"]})
topo = pd.DataFrame(rows)
tp, tn = topo.has_fb_pos_feedback == 1, topo.has_fb_neg_feedback == 1
topo["category"] = np.select([tp & ~tn, tn & ~tp, tp & tn],
                             ["PFL-only", "NFL-only", "Mixed"], default="No feedback")
topo["fb_group5"] = feedback_group5(topo)
gmap = topo.loc[(topo.disjoint == 0) & topo.fb_group5.isin(GROUP5),
                ["net_id", "fb_group5"]]

# --- traversal rate per set, then per network, at every level ----------------
COLS = (["net_id", "accepted_nodes", "stability_time"]
        + [f"mean_path_{g}" for g in GENES]
        + [f"gmm_mean{i}_{g}" for g in GENES for i in (1, 2)])

per_med, per_set, per_rcv = {}, {}, {}
n_med, n_set, n_rcv = {}, {}, {}
long_rows = []
for l in LEVELS:
    d = pd.read_csv(NOISE_FILES[l], usecols=COLS)
    P = np.stack([d[f"mean_path_{g}"].values for g in GENES], axis=1)
    lev = np.stack([d[[f"gmm_mean1_{g}", f"gmm_mean2_{g}"]].mean(axis=1).values
                    for g in GENES], axis=1)
    P = np.where(lev > 1e-9, P / lev, np.nan)
    acc = np.stack([d.accepted_nodes.str.contains(g).values for g in GENES], axis=1)
    with np.errstate(invalid="ignore"):
        v = np.nanmean(np.where(acc, P, np.nan), axis=1)
    rate = np.where(d.stability_time >= MIN_SETTLING,
                    PATH_RATE_SCALE * v / d.stability_time, np.nan)

    s = pd.DataFrame({"net_id": d.net_id.values, "rate": rate}).dropna().merge(gmap, on="net_id")
    agg = (s.groupby(["net_id", "fb_group5"]).rate
             .agg(n="size", med="median", q1=lambda x: x.quantile(0.25),
                  q3=lambda x: x.quantile(0.75)).reset_index())
    agg["rcv"] = (agg.q3 - agg.q1) / agg.med

    per_med[l] = {g: agg.loc[agg.fb_group5 == g, "med"].values for g in GROUP5}
    per_set[l] = {g: s.loc[s.fb_group5 == g, "rate"].values for g in GROUP5}
    sub = agg[(agg.n >= MIN_RUNS) & agg.rcv.replace(0, np.nan).notna()]
    per_rcv[l] = {g: sub.loc[sub.fb_group5 == g, "rcv"].values for g in GROUP5}

    n_med[l] = {g: len(per_med[l][g]) for g in GROUP5}
    n_rcv[l] = {g: len(per_rcv[l][g]) for g in GROUP5}
    n_set[l] = {g: len(per_set[l][g]) for g in GROUP5}
    long_rows.append(agg.assign(noise=l))

    print(f"sigma {l:<4g} {len(s):>8,} sets / {agg.net_id.nunique():>5,} networks   "
          f"median rate {np.median(s.rate):6.2f}   median per-network rCV "
          f"{np.median(sub.rcv):.3f} ({len(sub):,} networks with >= {MIN_RUNS} sets)")

long = pd.concat(long_rows)[["noise", "net_id", "fb_group5", "n", "med", "rcv"]]
long.rename(columns={"med": "traversal_median"}).to_csv(
    f"{RES}/nz_traversal_per_network{NZ_SUFFIX}.csv", index=False)

NMAX = max(n_med[l][g] for l in LEVELS for g in GROUP5)
SETMAX = max(n_set[l][g] for l in LEVELS for g in GROUP5)


def _kde_norm(v, logy=False):
    vv = np.random.default_rng(0).choice(v, KDE_CAP, replace=False) if len(v) > KDE_CAP else v
    vv = np.log10(vv) if logy else vv
    return lambda yy: (lambda dn: dn / dn.max())(gaussian_kde(vv)(yy))


def ridge_panel(ax, per, counts, ylabel, ymin=None, ymax=None, logy=False, cmax=None):
    allv = np.concatenate([per[l][g] for l in LEVELS for g in GROUP5])
    if ymin is None:
        ymin = np.quantile(allv, 0.002) * 0.9
    if ymax is None:
        ymax = np.quantile(allv, 0.998) * 1.05
    grid = (np.linspace(np.log10(ymin), np.log10(ymax), 240) if logy
            else np.linspace(ymin, ymax, 240))
    yy = 10 ** grid if logy else grid
    for i, l in enumerate(LEVELS):
        for g in GROUP5:
            v = per[l][g]
            if len(v) < 5:
                continue
            w = W * _kde_norm(v, logy)(grid)
            ax.fill_betweenx(yy, i, i + w, color=G5_COLOR[g], alpha=ALPHA, lw=0, zorder=2)
            ax.plot(i + w, yy, color=G5_COLOR[g], lw=0.9, alpha=0.6, zorder=3)

    for g in GROUP5:                                  # median trend, lines first
        ys = [np.median(per[l][g]) if len(per[l][g]) else np.nan for l in LEVELS]
        ax.plot(XPOS, ys, "-", color=G5_COLOR[g], lw=2.2,
                zorder=12 if g == "Pure Positive" else 10)
    cm = cmax or NMAX
    pts = [(i, float(np.median(per[l][g])), counts[l][g], g)
           for g in GROUP5 for i, l in enumerate(LEVELS) if len(per[l][g])]
    for xx, y, n, g in sorted(pts, key=lambda p: -p[2]):   # largest first
        # Pure Positive on top, as in nz_noise_panels: it is the category the noise
        # story turns on and its points crowd the others
        ax.scatter(xx, y, s=max(SMAX * n / cm, 12.0), color=G5_COLOR[g], edgecolor="white",
                   linewidth=1.0, zorder=14 if g == "Pure Positive" else 13)

    if logy:
        ax.set_yscale("log")
        ax.set_yticks([0.1, 0.2, 0.3, 0.5, 1.0])
        ax.set_yticklabels(["0.1", "0.2", "0.3", "0.5", "1.0"])
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(*XLIM_RIDGE)
    ax.set_xticks(XPOS)
    ax.set_xticklabels([f"{l:g}" for l in LEVELS], fontsize=TICK)
    ax.set_xlabel("Noise Level (a.u.)", fontsize=LAB)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LAB)
    ax.tick_params(labelsize=TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#EEEEEE", lw=0.8)


fig = plt.figure(figsize=(5.0, 3.2))
gs = fig.add_gridspec(1, 2, wspace=0.10)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharey=ax0)

allrate = np.concatenate([per_med[l][g] for l in LEVELS for g in GROUP5] +
                         [per_set[l][g] for l in LEVELS for g in GROUP5])
rmin, rmax = np.quantile(allrate, 0.002) * 0.9, np.quantile(allrate, 0.998) * 1.05
ridge_panel(ax0, per_med, n_med, METRIC2["axis"], ymin=rmin, ymax=rmax)
ridge_panel(ax1, per_set, n_set, None, ymin=rmin, ymax=rmax, cmax=SETMAX)
ax1.tick_params(labelleft=False)

for ax, txt in [(ax0, "per-network median"), (ax1, "all parameter sets")]:
    ax.text(0.03, 0.03, txt, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=TICK - 1.5, style="italic",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))

handles = [Line2D([0], [0], color=G5_COLOR[g], lw=2.2, marker="o", markeredgecolor="white",
                  markersize=7, label=g) for g in GROUP5]
leg_cat = ax0.legend(handles=handles, fontsize=LEG - 1, loc="upper left", frameon=False,
                     handlelength=1.3, handletextpad=0.5, labelspacing=0.25,
                     borderaxespad=0.2)
ax0.add_artist(leg_cat)          # kept when the size legend is added to the same axes
# two size references, not three, and a short title: at 2.1 inches per panel a
# three-entry legend is wider than the axes and spills into the neighbour
ax1.legend(handles=[plt.scatter([], [], s=max(SMAX * n / SETMAX, 12.0), color="#888",
                                edgecolor="white", linewidth=1.0, label=f"{n // 1000}k")
                    for n in (25000, 175000)],
           fontsize=LEG - 2, loc="lower right", frameon=True, facecolor="white",
           framealpha=0.85, edgecolor="none", title="Sets", title_fontsize=LEG - 2,
           labelspacing=0.7, borderpad=0.4, handletextpad=0.6, borderaxespad=0.2,
           scatterpoints=1)
ax0.legend(handles=[plt.scatter([], [], s=max(SMAX * n / NMAX, 12.0), color="#888",
                                edgecolor="white", linewidth=1.0, label=str(n))
                    for n in (100, 900)],
           fontsize=LEG - 2, loc="lower right", frameon=True, facecolor="white",
           framealpha=0.85, edgecolor="none", title="Nets", title_fontsize=LEG - 2,
           labelspacing=0.7, borderpad=0.4, handletextpad=0.6, borderaxespad=0.2,
           scatterpoints=1)

savefig(f"{RES}/nz_traversal_panels{NZ_SUFFIX}", bbox_inches="tight")
print(f"\n[set: {NOISE_SET}, levels {' / '.join(str(l) for l in LEVELS)}]  "
      f"Saved -> nz_traversal_panels{NZ_SUFFIX}.{{png,svg}} + "
      f"nz_traversal_per_network{NZ_SUFFIX}.csv")
