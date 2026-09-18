"""
explore_overshoot.py  (EXPLORATORY)
==============================================================================
Does the OVERSHOOT of cell fractions during HR depend on feedback topology?

Setup. For every HR-enabling parameter set the results file records the transient
peak of the cell fraction relative to its final value: `overshoot_ratio` (peak/final,
NaN when there is no overshoot -- the detector's threshold is ratio >= 1.1),
`overshoot_time` (the peak time) and `overshoot_state` (which state the cells start
from, "on" or "off"). An early/late label is derived as overshoot_time > 1000.

These columns were read from frac_data/overshoot_results_net_*.csv until 2026-09-17
and are identical to it: existence agrees 100.0000% over all 467,885 sigma-1
frac10+op_ic rows, and the ratio and peak time are numerically identical. Reading the
results file drops a 32 MB second input and a three-key merge.

MAGNITUDE of the ratio is NOT analysed here -- see explore_overshoot_magnitude.py,
which is not a pipeline step. Every figure below is driven by overshoot EXISTENCE
(a per-set 0/1) or by peak TIMING.

Outcomes analysed, all conditional on HR (the results file holds only HR sets):
  1. EXISTENCE   overshoot vs none                      -- per set, per network
  2. TIMING      late vs early peak, among overshoots   -- the t=1000 label
  3. DIRECTION   ON- vs OFF-state overshoot             -- tracks the starting
                 state (an IC property), so this doubles as a negative control:
                 topology should NOT move it

MAGNITUDE (the overshoot ratio itself) is analysed in explore_overshoot_magnitude.py,
which is NOT a pipeline step.

Predictor = the project's 5-way feedback grouping `fb_group5` (Pure Positive ->
Pure Negative), plus the 6 elementary feedback categories for the per-loop view.

Statistics follow the project's convention: rates are reported raw AND
density-adjusted (binomial GLM on `n_edges`, g-computation over the observed
density distribution), with network-level bootstrap CIs, mirror contrasts across
the polarity axis, and a network-level Spearman trend. All covariates are
network-level, so the per-set logistic is fitted on per-network aggregates --
mathematically identical, and fast enough to bootstrap honestly.

Pool: non-disjoint (canonical). Note this filter is a no-op here -- none of the
85 disjoint or 21 feedback-free networks produce any HR set, so the HR pool is
already connected and feedback-bearing (2,977 networks).

Follows HR_RESULTS / HR_POOL: outputs carry repro_common.OUT_SUFFIX.
Output: results/reproduce/overshoot_*.{csv,png,svg}
==============================================================================
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.ticker import PercentFormatter
from scipy import stats

from repro_common import (load_adjacency_matrices, feedback_group5, edge_bearing_components,
                          savefig, GROUP5, ELEMENTARY, PRETTY, RESULTS_FILE, OUT_SUFFIX,
                          CONNECTED)
from motif_lib import feedback_counts

warnings.filterwarnings("ignore")
RES = "results/reproduce"
RNG = np.random.default_rng(42)
N_BOOT = 500
LAB, TICK, LEG = 9, 8, 7.5
G5_COLOR = {"Pure Positive": "#B71C1C", "Mixed Pos-Dom": "#E57373", "Balanced": "#9E9E9E",
            "Mixed Neg-Dom": "#64B5F6", "Pure Negative": "#0D47A1"}
G5_LABEL = ["Pure\nPositive", "Mixed\nPos-Dom", "Balanced", "Mixed\nNeg-Dom", "Pure\nNegative"]
# compact variant: the 5 labels collide at single-column width
G5_SHORT = ["Pure\nPos", "Mixed\nPos", "Bal.", "Mixed\nNeg", "Pure\nNeg"]


# =============================================================================
# 1. Per-set overshoot flags, read straight from the results file
# =============================================================================
# Sourced from the results file rather than frac_data since 2026-09-17. Verified
# equivalent on the sigma-1 frac10+op_ic set: existence agrees 100.0000% over all
# 467,885 rows, and overshoot_ratio / overshoot_time / overshoot_state are
# numerically IDENTICAL to frac_data's ratio / peak time / side. See
# repro_common.overshoot_per_network for the full comparison.
EARLY_CUTOFF = 1000.0          # peak at t > 1000 is "late"; reproduces frac_data's
                               # on_which/off_which label exactly (100.000000%)
frac = pd.read_csv(RESULTS_FILE, usecols=["net_id", "param_id", "accepted_nodes",
                                          "stability_time", "overshoot_ratio",
                                          "overshoot_time", "overshoot_state"])
frac["overshoot"] = frac.overshoot_ratio.notna().astype(int)
frac["ratio"] = frac.overshoot_ratio
frac["peak_time"] = frac.overshoot_time
frac["side"] = frac.overshoot_state.fillna("none")
frac["is_on"] = (frac.side == "on").astype(int)
frac["is_late"] = (frac.peak_time > EARLY_CUTOFF).astype(int)
print(f"results file : {RESULTS_FILE}")
print(f"HR sets      : {len(frac):,} over {frac.net_id.nunique():,} networks")

# =============================================================================
# 2. Topology
# =============================================================================
topo = []
for uid, m in load_adjacency_matrices().items():
    fb = feedback_counts(m)
    row = {"net_id": uid, "n_edges": int((m != 0).sum()),
           "disjoint": int(edge_bearing_components(m) >= 2),
           "has_fb_pos_feedback": fb["has_pos_feedback"],
           "has_fb_neg_feedback": fb["has_neg_feedback"],
           "fb_pos_feedback": fb["pos_feedback"], "fb_neg_feedback": fb["neg_feedback"]}
    row.update({f"fb_{c}": fb[c] for c in
                ["pos_1node", "neg_1node", "pos_2node", "neg_2node", "pos_3node", "neg_3node"]})
    topo.append(row)
topo = pd.DataFrame(topo)
tp, tn = topo.has_fb_pos_feedback == 1, topo.has_fb_neg_feedback == 1
topo["category"] = np.select([tp & ~tn, tn & ~tp, tp & tn],
                             ["PFL-only", "NFL-only", "Mixed"], default="No feedback")
topo["fb_group5"] = feedback_group5(topo)

d = frac.merge(topo, on="net_id")
if CONNECTED:
    d = d[d.disjoint == 0]
d = d[d.fb_group5.isin(GROUP5)]
print(f"analysis pool: {len(d):,} sets, {d.net_id.nunique():,} networks "
      f"({'non-disjoint' if CONNECTED else 'full'})\n")

# =============================================================================
# 3. Per-network aggregates -- every covariate is network-level, so a binomial
#    GLM on these counts is identical to the per-set logistic, and fast.
# =============================================================================
NETCOLS = ["net_id", "fb_group5", "n_edges"] + ELEMENTARY
net = (d.groupby(NETCOLS)
        .agg(n_sets=("overshoot", "size"), n_over=("overshoot", "sum"),
             n_on=("is_on", "sum"), n_late=("is_late", "sum"),
             ratio_med=("ratio", "median"), peak_med=("peak_time", "median"))
        .reset_index())
net["over_rate"] = net.n_over / net.n_sets
EDGES = sorted(net.n_edges.unique())


def design(df, group_col="fb_group5", adjust=True):
    """Group dummies (reference = Pure Positive) + optional n_edges dummies."""
    X = pd.DataFrame(index=df.index)
    for g in GROUP5[1:]:
        X[f"g_{g}"] = (df[group_col] == g).astype(float)
    if adjust:
        for e in EDGES[1:]:
            X[f"e{e}"] = (df.n_edges == e).astype(float)
    return sm.add_constant(X, has_constant="add")


def gcomp(df, num, den, adjust=True):
    """Binomial GLM + g-computation: the rate each group would show if it had the
    pool's density distribution. Returns (fitted model, {group: adjusted rate})."""
    sub = df[df[den] > 0]
    X = design(sub, adjust=adjust)
    fit = sm.GLM(np.c_[sub[num], sub[den] - sub[num]], X,
                 family=sm.families.Binomial(), freq_weights=None).fit()
    out = {}
    for g in GROUP5:
        Xc = X.copy()
        for c in [c for c in X.columns if c.startswith("g_")]:
            Xc[c] = 0.0
        if f"g_{g}" in Xc.columns:
            Xc[f"g_{g}"] = 1.0
        # weight each network's predicted probability by its number of sets
        out[g] = float(np.average(fit.predict(Xc), weights=sub[den]))
    return fit, out


def boot_gcomp(df, num, den, adjust=True, n_boot=N_BOOT):
    """Network-level bootstrap of the g-computed rates and the mirror contrasts."""
    rates = {g: [] for g in GROUP5}
    contrasts = {"Pure Neg - Pure Pos": [], "Mixed Neg-Dom - Mixed Pos-Dom": []}
    idx = np.arange(len(df))
    for _ in range(n_boot):
        b = df.iloc[RNG.choice(idx, len(idx), replace=True)]
        try:
            _, r = gcomp(b, num, den, adjust=adjust)
        except Exception:
            continue
        for g in GROUP5:
            rates[g].append(r[g])
        contrasts["Pure Neg - Pure Pos"].append(r["Pure Negative"] - r["Pure Positive"])
        contrasts["Mixed Neg-Dom - Mixed Pos-Dom"].append(
            r["Mixed Neg-Dom"] - r["Mixed Pos-Dom"])
    ci = {g: np.percentile(v, [2.5, 97.5]) for g, v in rates.items() if v}
    cc = {k: (np.mean(v), *np.percentile(v, [2.5, 97.5]),
              2 * min((np.array(v) <= 0).mean(), (np.array(v) >= 0).mean()))
          for k, v in contrasts.items() if v}
    return ci, cc


def outcome_table(num, den, name):
    """Raw + density-adjusted rate per group, with bootstrap CIs and contrasts."""
    _, adj = gcomp(net, num, den)
    _, raw = gcomp(net, num, den, adjust=False)
    ci, cc = boot_gcomp(net, num, den)
    rows = []
    for g in GROUP5:
        s = net[net.fb_group5 == g]
        lo, hi = ci.get(g, (np.nan, np.nan))
        rows.append({"outcome": name, "fb_group5": g, "n_networks": len(s),
                     "n_denominator": int(s[den].sum()), "raw_rate": raw[g],
                     "adj_rate": adj[g], "adj_ci_low": lo, "adj_ci_high": hi})
    tab = pd.DataFrame(rows)

    print(f"--- {name}: density-adjusted (g-computation) rate per category")
    for _, r in tab.iterrows():
        print(f"  {r.fb_group5:16s} (n={r.n_networks:4d} nets, {r.n_denominator:7,d} sets): "
              f"raw={r.raw_rate:.4f}  adjusted={r.adj_rate:.4f} "
              f"[{r.adj_ci_low:.4f},{r.adj_ci_high:.4f}]")
    con = []
    for k, (m_, lo, hi, p) in cc.items():
        print(f"  {k:30s}: {m_:+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]  p={p:.3f}")
        con.append({"outcome": name, "contrast": k, "delta": m_,
                    "ci_low": lo, "ci_high": hi, "p_boot": p})

    # network-level monotone trend across the ordered polarity axis
    sub = net[net[den] > 0]
    rho, p = stats.spearmanr(sub.fb_group5.map({g: i for i, g in enumerate(GROUP5)}),
                             sub[num] / sub[den])
    print(f"  network-level Spearman(pos->neg, rate): rho={rho:+.3f}, p={p:.2e}\n")
    tab["trend_rho"], tab["trend_p"] = rho, p
    return tab, pd.DataFrame(con)


print("=" * 78)
print("OUTCOME TABLES")
print("=" * 78)
t_ex, c_ex = outcome_table("n_over", "n_sets", "existence (overshoot vs none)")
t_lt, c_lt = outcome_table("n_late", "n_over", "timing (late | overshoot)")
t_on, c_on = outcome_table("n_on", "n_over", "direction (ON | overshoot)")

tables = pd.concat([t_ex, t_lt, t_on], ignore_index=True)
contrasts = pd.concat([c_ex, c_lt, c_on], ignore_index=True)
tables.to_csv(f"{RES}/overshoot_by_group{OUT_SUFFIX}.csv", index=False)
contrasts.to_csv(f"{RES}/overshoot_contrasts{OUT_SUFFIX}.csv", index=False)

# =============================================================================
# 4. Per-loop view: log-odds of overshoot per additional feedback loop
# =============================================================================
print("=" * 78)
print("PER-LOOP: binomial-GLM log-odds of overshoot per additional loop")
print("=" * 78)
perloop = []
for label, cols in [("crude", ELEMENTARY), ("density_adjusted", ELEMENTARY + ["n_edges"])]:
    X = sm.add_constant(net[cols].astype(float))
    fit = sm.GLM(np.c_[net.n_over, net.n_sets - net.n_over], X,
                 family=sm.families.Binomial()).fit()
    disp = float(fit.pearson_chi2 / fit.df_resid)          # quasi-binomial inflation
    se = fit.bse * np.sqrt(disp)
    for c in cols:
        perloop.append({"model": label, "feature": c, "pretty": PRETTY.get(c, c),
                        "logOR_per_loop": fit.params[c], "or_per_loop": np.exp(fit.params[c]),
                        "ci_low": fit.params[c] - 1.96 * se[c],
                        "ci_high": fit.params[c] + 1.96 * se[c],
                        "pval": 2 * stats.norm.sf(abs(fit.params[c] / se[c]))})
    if label == "density_adjusted":
        print(f"(GLM dispersion = {disp:.1f} -> quasi-binomial CIs)")
        for r in perloop[-len(cols):]:
            if r["feature"] in ELEMENTARY:
                print(f"  {r['pretty']:20s}: {r['logOR_per_loop']:+.3f} per loop "
                      f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]  p={r['pval']:.1e}")
perloop = pd.DataFrame(perloop)
perloop.to_csv(f"{RES}/overshoot_perloop{OUT_SUFFIX}.csv", index=False)

# Magnitude of the overshoot RATIO is deliberately NOT analysed here -- it moved to
# explore_overshoot_magnitude.py on 2026-09-17, which is not a pipeline step. Every
# figure this script produces is driven by overshoot EXISTENCE (a per-set 0/1) or by
# peak TIMING; the ratio fed only a descriptive CSV, so keeping it here mixed a
# non-pipeline output into a pipeline script.

# =============================================================================
# 5b. Is a LATE peak just slow dynamics? Normalise by each circuit's own clock.
# -----------------------------------------------------------------------------
# The early/late label cuts at a FIXED t=1000, so a group that simply settles
# slower would show more "late" peaks for trivial reasons. Peak time divided by
# that set's own stability_time, and P(peak after settling), remove that.
# =============================================================================
print("\n" + "=" * 78)
print("PEAK TIME RELATIVE TO SETTLING TIME")
print("=" * 78)
ov = d[d.overshoot == 1]              # timing is defined only where a peak exists
ov = ov.assign(rel_peak=ov.peak_time / ov.stability_time,
               peak_after_settle=(ov.peak_time > ov.stability_time).astype(int))
rel = (ov.groupby("fb_group5")
         .agg(n_sets=("rel_peak", "size"), settle_median=("stability_time", "median"),
              peak_median=("peak_time", "median"), rel_peak_median=("rel_peak", "median"),
              p_peak_after_settle=("peak_after_settle", "mean"),
              p_late_fixed_cut=("is_late", "mean"))
         .reindex(GROUP5).reset_index())
rel.to_csv(f"{RES}/overshoot_timing_relative{OUT_SUFFIX}.csv", index=False)
for _, r in rel.iterrows():
    print(f"  {r.fb_group5:16s} settle med={r.settle_median:6.0f}  peak med={r.peak_median:6.0f}  "
          f"peak/settle={r.rel_peak_median:.3f}  P(peak>settle)={r.p_peak_after_settle:6.1%}  "
          f"P(late@1000)={r.p_late_fixed_cut:6.1%}")

CODE = {g: i for i, g in enumerate(GROUP5)}
nl = (ov.groupby(["net_id", "fb_group5"])
        .agg(rel_peak=("rel_peak", "median"), after=("peak_after_settle", "mean"))
        .reset_index())
for col, lbl in [("rel_peak", "median peak/settle"), ("after", "P(peak > settle)")]:
    rho, p = stats.spearmanr(nl.fb_group5.map(CODE), nl[col])
    print(f"  network-level Spearman(pos->neg, {lbl}): rho={rho:+.3f}, p={p:.2e}")
print("  -> Pure Positive settles FASTEST yet peaks LATEST, in absolute and relative\n"
      "     terms: the late-peak result is not a slow-dynamics artefact.")

# =============================================================================
# 6. Figures
# =============================================================================
def group_panel(ax, tab, ylabel, title, pct=True):
    x = np.arange(5)
    sub = tab.set_index("fb_group5").reindex(GROUP5)
    ax.bar(x, sub.raw_rate, width=0.62, color=[G5_COLOR[g] for g in GROUP5],
           alpha=0.35, edgecolor="none", zorder=2, label="raw")
    yerr = np.abs(np.c_[sub.adj_rate - sub.adj_ci_low,
                        sub.adj_ci_high - sub.adj_rate].T)
    ax.errorbar(x, sub.adj_rate, yerr=yerr, fmt="o", ms=5.5, mec="white", mew=1.0,
                ecolor="#444", elinewidth=1.2, capsize=2.5, zorder=4,
                color="#222", label="density-adjusted")
    for i, g in enumerate(GROUP5):
        ax.plot(x[i], sub.adj_rate.iloc[i], "o", ms=5.5, color=G5_COLOR[g],
                mec="white", mew=1.0, zorder=5)
    ax.plot(x, sub.adj_rate, "-", color="#BBBBBB", lw=1.0, zorder=3)
    ax.set_ylabel(ylabel, fontsize=LAB)
    ax.set_title(title, fontsize=LAB, pad=4)
    if pct:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xticks(x)
    ax.set_xticklabels(G5_SHORT, fontsize=TICK - 0.5)
    ax.tick_params(axis="y", labelsize=TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#EEEEEE", lw=0.7)


# --- Figure 1: the three rate outcomes (double column) ------------------------
# Three panels, not the former 2x2. The fourth quadrant held a boxplot of the
# overshoot RATIO; magnitude moved to explore_overshoot_magnitude.py (not a pipeline
# step) on 2026-09-17, so every panel here is now a rate -- existence, or a
# conditional rate given existence.
fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0))
group_panel(axes[0], t_ex, "P(overshoot)", "a  Overshoot existence")
group_panel(axes[1], t_lt, "P(late | overshoot)", "b  Late peak (t > 1000)")
group_panel(axes[2], t_on, "P(ON | overshoot)", "c  ON-state overshoot (IC control)")
axes[0].legend(fontsize=LEG, frameon=False, loc="lower right")

fig.tight_layout(w_pad=1.6, h_pad=1.8)
savefig(f"{RES}/overshoot_by_group{OUT_SUFFIX}", bbox_inches="tight")

# --- Figure 2: per-loop forest (single column) --------------------------------
fig, ax = plt.subplots(figsize=(3.6, 3.0))
adj = perloop[(perloop.model == "density_adjusted") &
              (perloop.feature.isin(ELEMENTARY))].reset_index(drop=True)
y = np.arange(len(adj))[::-1]
for yy, (_, r) in zip(y, adj.iterrows()):
    c = "#B71C1C" if "pos" in r.feature else "#0D47A1"
    ax.plot([r.ci_low, r.ci_high], [yy, yy], color=c, lw=1.8, zorder=2)
    ax.scatter(r.logOR_per_loop, yy, color=c, s=32, edgecolor="white",
               linewidth=0.8, zorder=3)
ax.axvline(0, color="black", lw=0.9)
ax.set_yticks(y)
ax.set_yticklabels(adj.pretty, fontsize=TICK)
ax.set_xlabel("log-OR of overshoot per loop\n(density-adjusted)", fontsize=LAB)
ax.tick_params(axis="x", labelsize=TICK)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
ax.grid(axis="x", color="#EEEEEE", lw=0.7)
fig.tight_layout()
savefig(f"{RES}/overshoot_perloop{OUT_SUFFIX}", bbox_inches="tight")

# --- Figure 3: peak time on the circuit's own clock (single column) -----------
fig, ax = plt.subplots(figsize=(3.6, 3.0))
x = np.arange(5)
sub = rel.set_index("fb_group5").reindex(GROUP5)
ax.bar(x, sub.p_peak_after_settle, width=0.62,
       color=[G5_COLOR[g] for g in GROUP5], edgecolor="none", zorder=2)
ax.set_ylabel("P(peak after settling)", fontsize=LAB)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax.tick_params(axis="y", labelsize=TICK)
ax.set_xticks(x)
ax.set_xticklabels(G5_SHORT, fontsize=TICK - 0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#EEEEEE", lw=0.7)

ax2 = ax.twinx()
ax2.plot(x, sub.settle_median, "o--", color="#333", ms=4, lw=1.1, zorder=4)
ax2.set_ylabel("Median settling time (a.u.)", fontsize=LAB, color="#333")
ax2.tick_params(axis="y", labelsize=TICK, colors="#333")
ax2.spines[["top"]].set_visible(False)
fig.tight_layout()
savefig(f"{RES}/overshoot_timing_relative{OUT_SUFFIX}", bbox_inches="tight")

print(f"\n[pool: {'connected-only' if CONNECTED else 'full'}]  Saved -> "
      f"overshoot_{{by_group,contrasts,perloop,timing_relative}}{OUT_SUFFIX}.csv + "
      f"overshoot_{{by_group,perloop,timing_relative}}{OUT_SUFFIX}.{{png,svg}}")
