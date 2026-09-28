"""
visualize_noise_panels.py
==============================================================================
The four noise-thread KDE ridges in one 2x2 grid:
  top-left     HR Robustness              per-network rate          + MEDIAN trend
  top-right    Settling-time variability  per-network rCV, LOG y    + MEDIAN trend
  bottom-left  HR Settling Time           per-network median        + MEDIAN trend
  bottom-right HR Settling Time           all parameter sets        + MEDIAN trend

Layout is unchanged from the original version of this script. What differs:

  * NOISE_SET-aware (repro_common): levels, input files and the output suffix all
    follow the selected set, so this runs on frac10_opic (sigma 0.2 / 1.0 / 5.0)
    as well as the legacy sigma 0.2 / 1.0 / 2.0.
  * MEDIAN everywhere. The robustness panel previously threaded the MEAN. The two
    disagree sharply: Pure Positive at sigma 5 has mean 0.105 (highest of any
    category) but median 0.005 (lowest), because its mean is carried by ~54
    mutual-repression circuits in the tail. Median matches every other violin/KDE
    figure in the project and describes the typical circuit.
  * rCV on a LOG y-axis, with its KDE fitted on log10(rCV) and mapped back -- rCV
    is a right-skewed ratio spanning ~1.3 decades, and a linear-space density drawn
    on a log axis misplaces its own mode.
  * Settling-time panels floored at 500 rather than at the global minimum, so the
    bulk (1200-1900) is not squeezed by a handful of very fast circuits.
  * MARKER AREA IS PROPORTIONAL TO THE NUMBER OF NETWORKS behind each point. The
    counts differ a lot between categories and levels -- the rCV panel additionally
    requires >= MIN_RUNS sets per network, which removes far more Pure Positive
    circuits than Pure Negative ones -- so a fixed marker would hide how much each
    median rests on. Counts are NETWORKS in every panel, including the
    all-parameter-set one, where they are the networks contributing sets.

Output: results/reproduce/nz_noise_panels<suffix>.{png,svg}
==============================================================================
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from repro_common import savefig, GROUP5, NOISE_LEVELS, NOISE_FILES, NZ_SUFFIX, NOISE_SET

warnings.filterwarnings("ignore")
RES = "results/reproduce"
os.makedirs(RES, exist_ok=True)

PANEL_W, PANEL_H = 5.0, 4.8
LAB, TICK, LEG = 14, 12, 12
W = 0.80                       # one-sided KDE width (extends right)
KDE_CAP = 40000                # subsample cap for the run-level KDE
ALPHA = 0.20
ST_FLOOR = 500.0               # settling-time axis floor
SMAX = 320.0                   # marker area for the largest network count
LEVELS = NOISE_LEVELS
XPOS = range(len(LEVELS))
XLIM_RIDGE = (-0.35, len(LEVELS) - 1 + W + 0.12)
G5_COLOR = {"Pure Positive": "#B71C1C", "Mixed Pos-Dom": "#E57373", "Balanced": "#9E9E9E",
            "Mixed Neg-Dom": "#64B5F6", "Pure Negative": "#0D47A1"}

# --- data -------------------------------------------------------------------
rn = pd.read_csv(f"{RES}/nz_rate_per_network{NZ_SUFFIX}.csv")
pn = pd.read_csv(f"{RES}/nz_settling_per_network{NZ_SUFFIX}.csv")
sv = pd.read_csv(f"{RES}/nz_stvar_per_network{NZ_SUFFIX}.csv")
gmap = rn[["net_id", "fb_group5"]].drop_duplicates()

per_rate = {l: {g: rn[(rn.noise == l) & (rn.fb_group5 == g)]["rate"].values
                for g in GROUP5} for l in LEVELS}
per_stn = {l: {g: pn[(pn.noise == l) & (pn.fb_group5 == g)]["st_median"].dropna().values
               for g in GROUP5} for l in LEVELS}
per_rcv = {l: {g: sv[(sv.noise == l) & (sv.fb_group5 == g)]["rcv"].replace(0, np.nan)
               .dropna().values for g in GROUP5} for l in LEVELS}

per_run, run_nets = {}, {}          # all-parameter-set settling times + their network counts
for l in LEVELS:
    d = pd.read_csv(NOISE_FILES[l], usecols=["net_id", "stability_time"]).merge(gmap, on="net_id")
    per_run[l] = {g: d.loc[d.fb_group5 == g, "stability_time"].dropna().values for g in GROUP5}
    run_nets[l] = {g: int(d.loc[d.fb_group5 == g, "net_id"].nunique()) for g in GROUP5}

# network count behind every point, per panel
n_rate = {l: {g: len(per_rate[l][g]) for g in GROUP5} for l in LEVELS}
n_stn = {l: {g: len(per_stn[l][g]) for g in GROUP5} for l in LEVELS}
n_rcv = {l: {g: len(per_rcv[l][g]) for g in GROUP5} for l in LEVELS}
NMAX = max(max(d[l][g] for l in LEVELS for g in GROUP5)
           for d in (n_rate, n_stn, n_rcv, run_nets))


def _kde_norm(v, logy=False):
    vv = np.random.default_rng(0).choice(v, KDE_CAP, replace=False) if len(v) > KDE_CAP else v
    vv = np.log10(vv) if logy else vv
    return lambda yy: (lambda dn: dn / dn.max())(gaussian_kde(vv)(yy))


def ridge_panel(ax, per, counts, ylabel, pct=False, ymin=None, ymax=None, logy=False,
                cmax=None):
    allv = np.concatenate([per[l][g] for l in LEVELS for g in GROUP5])
    if ymin is None:
        ymin = 0.0 if pct else allv.min() * 0.98
    if ymax is None:
        ymax = np.quantile(allv, 0.98) if pct else allv.max() * 1.02
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
        ax.plot(XPOS, ys, "-", color=G5_COLOR[g], lw=2.6,
                zorder=12 if g == "Pure Positive" else 10)
    cm = cmax or NMAX
    pts = [(i, float(np.median(per[l][g])), counts[l][g], g)
           for g in GROUP5 for i, l in enumerate(LEVELS) if len(per[l][g])]
    for x, y, n, g in sorted(pts, key=lambda p: -p[2]):   # largest first, so small stay visible
        # Pure Positive on top: it is the category the noise story turns on, and its
        # points sit close to Pure Negative's near the floor of the robustness panel
        ax.scatter(x, y, s=max(SMAX * n / cm, 14.0), color=G5_COLOR[g], edgecolor="white",
                   linewidth=1.1, zorder=14 if g == "Pure Positive" else 13)

    if logy:
        ax.set_yscale("log")
        ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
        ax.set_yticklabels(["0.05", "0.1", "0.2", "0.3", "0.5", "1.0"])
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(ymin, ymax)
    if pct:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    _finish(ax, ylabel, XLIM_RIDGE)


def _finish(ax, ylabel, xlim):
    ax.set_xlim(*xlim)
    ax.set_xticks(XPOS)
    ax.set_xticklabels([str(l) for l in LEVELS], fontsize=TICK)
    ax.set_xlabel("Noise Level (a.u.)", fontsize=LAB)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LAB)
    ax.tick_params(labelsize=TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#EEEEEE", lw=0.8)


fig = plt.figure(figsize=(2 * PANEL_W, 2 * PANEL_H))
top, bot = fig.subfigures(2, 1, hspace=0.03)
ax00, ax01 = top.subplots(1, 2, gridspec_kw={"wspace": 0.30})
ax10, ax11 = bot.subplots(1, 2, sharey=True, gridspec_kw={"wspace": 0.06})

ridge_panel(ax00, per_rate, n_rate, "HR Robustness", pct=True)
allrcv = np.concatenate([per_rcv[l][g] for l in LEVELS for g in GROUP5])
ridge_panel(ax01, per_rcv, n_rcv, "rCV of Settling Time\n(IQR / median)",
            ymin=np.quantile(allrcv, 0.002) * 0.85,
            ymax=np.quantile(allrcv, 0.998) * 1.15, logy=True)

allbot = np.concatenate([per_stn[l][g] for l in LEVELS for g in GROUP5] +
                        [per_run[l][g] for l in LEVELS for g in GROUP5])
bmax = allbot.max() * 1.02
ridge_panel(ax10, per_stn, n_stn, "HR Settling Time (a.u.)", ymin=ST_FLOOR, ymax=bmax)
n_run_sets = {l: {g: len(per_run[l][g]) for g in GROUP5} for l in LEVELS}
SETMAX = max(n_run_sets[l][g] for l in LEVELS for g in GROUP5)
ridge_panel(ax11, per_run, n_run_sets, None, ymin=ST_FLOOR, ymax=bmax, cmax=SETMAX)
ax11.tick_params(labelleft=False)
for ax, txt in [(ax10, "per-network median"), (ax11, "all parameter sets")]:
    ax.text(0.03, 0.03, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=TICK,
            style="italic", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))

handles = [Line2D([0], [0], color=G5_COLOR[g], lw=2.6, marker="o", markeredgecolor="white",
                  markersize=9, label=g) for g in GROUP5]
ax00.legend(handles=handles, fontsize=LEG, loc="upper left", frameon=False)
size_ref = [100, 400, 900]
ax01.legend(handles=[plt.scatter([], [], s=SMAX * n / NMAX, color="#888", edgecolor="white",
                                 linewidth=1.0, label=str(n)) for n in size_ref],
            fontsize=LEG - 2, loc="upper right", frameon=True, facecolor="white",
            framealpha=0.85, edgecolor="none", title="Networks", title_fontsize=LEG - 2,
            labelspacing=1.0, borderpad=0.7, handletextpad=1.0, scatterpoints=1)

ax11.legend(handles=[plt.scatter([], [], s=max(SMAX * n / SETMAX, 14.0), color="#888",
                                 edgecolor="white", linewidth=1.0, label=f"{n // 1000}k")
                     for n in (25000, 100000, 175000)],
            fontsize=LEG - 2, loc="upper right", frameon=True, facecolor="white",
            framealpha=0.85, edgecolor="none", title="Parameter sets",
            title_fontsize=LEG - 2, labelspacing=1.0, borderpad=0.7,
            handletextpad=1.0, scatterpoints=1)

savefig(f"{RES}/nz_noise_panels{NZ_SUFFIX}", bbox_inches="tight")
print(f"[set: {NOISE_SET}, levels {' / '.join(str(l) for l in LEVELS)}]  "
      f"network counts behind each median (min {min(min(n_rcv[l].values()) for l in LEVELS)}, "
      f"max {NMAX})")
print(f"Saved -> nz_noise_panels{NZ_SUFFIX}.{{png,svg}}")
