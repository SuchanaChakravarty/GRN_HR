"""
reproduce_noise.py
==============================================================================
Standalone reproduction of the noise-thread key results (Phases 15-19), using the
adopted 5-way feedback grouping (Pure Positive / Mixed Pos-Dom / Balanced /
Mixed Neg-Dom / Pure Negative). Depends only on the raw files:
    noise 0.2  multimodal_results_combined_sorted_noise_0o2.csv   (reduced)
    noise 1.0  multimodal_results_combined_sorted.csv             (baseline)
    noise 2.0  multimodal_results_combined_sorted_noise_2.csv     (increased)
plus network_data.json (topology via repro_common.build_feature_table).

Phase 16/17 — when noise drops 1.0 -> 0.2, ~68% of baseline-multimodal parameter
  sets (keyed (net_id, param_id)) are LOST. Do the lost sets differ from the
  retained ones in baseline settling time (16) or fluctuation (17)? Within-network
  paired median difference delta = median(lost) - median(retained) per network.
Phase 18 — per-network Spearman(multimodal_rate, median cv_before_x1) within each
  feedback group, raw and density-controlled (partial on n_edges).
Phase 19 — within-network variability of settling time (CV = std/mean, rCV =
  IQR/median) by feedback group across the three noise levels (the crossover).

Phase 15 — robustness (mean multimodal_rate) and settling-time level (median) per
  feedback group across the three noise levels (noise shapes both, polarity-split).

Outputs (results/reproduce/, NO figures -- see visualize_noise.py):
    nz_level_by_noise.csv                         Phase 15 (mean rate + median settling by group x noise)
    nz_settling_per_network.csv                   Phase 15 (per-network median settling; for the KDE ridge)
    nz_rate_per_network.csv                       Phase 15 (per-network robustness rate; for the KDE ridge)
    nz_loss_stability_pernet.csv / _summary.csv   Phase 16 (per-network delta + group test)
    nz_loss_fluct_pernet.csv     / _summary.csv   Phase 17
    nz_robflu_correlations.csv                    Phase 18 (raw + density-partial Spearman)
    nz_stvar_by_noise.csv                         Phase 19 (median CV & rCV by group x noise)
==============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

from repro_common import (build_feature_table, feedback_group5, spearman_ci, GROUP5,
                          NOISE_SET, NOISE_LEVELS, NOISE_FILES, NZ_SUFFIX,
                          load_adjacency_matrices, edge_bearing_components,
                          crude_and_mh)

warnings.filterwarnings("ignore")
os.makedirs("results/reproduce", exist_ok=True)

# Noise levels and their files come from repro_common.NOISE_SETS (env NOISE_SET).
LEVELS = NOISE_LEVELS
LOW, MID, HIGH = LEVELS[0], LEVELS[1], LEVELS[2]
BASELINE, REDUCED = NOISE_FILES[MID], NOISE_FILES[LOW]
LVLTXT = " / ".join(str(l) for l in LEVELS)
MIN_PER_GROUP = 5     # min lost & retained sets per network for the paired test
MIN_RUNS = 20         # min parameter sets per network for a variability estimate

# --- topology / grouping (noise-independent) ---------------------------------
feat = build_feature_table(results_file=BASELINE)
feat["fb_group5"] = feedback_group5(feat)
gmap = feat[["net_id", "fb_group5", "n_edges", "multimodal_rate"]]


def partial_spearman(x, y, z):
    """Rank partial correlation of x,y controlling z (+ approx p, df = n-3)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy, rxz, ryz = (np.corrcoef(rx, ry)[0, 1], np.corrcoef(rx, rz)[0, 1],
                     np.corrcoef(ry, rz)[0, 1])
    pr = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
    n = len(x)
    t = pr * np.sqrt((n - 3) / (1 - pr**2))
    return pr, 2 * stats.t.sf(abs(t), df=n - 3)


# =============================================================================
# Phase 16 & 17 — noise-reduction loss: lost vs retained, within-network paired
# =============================================================================
def loss_analysis(metric, tag):
    base = pd.read_csv(BASELINE, usecols=["net_id", "param_id", metric])
    low = pd.read_csv(REDUCED, usecols=["net_id", "param_id"])
    low["in_low"] = 1
    m = base.merge(low, on=["net_id", "param_id"], how="left")
    m["lost"] = m["in_low"].isna()
    m["retained"] = ~m["lost"]
    if metric != "stability_time":
        m = m.dropna(subset=[metric])
    m = m.merge(gmap[["net_id", "fb_group5"]], on="net_id", how="left")
    m = m[m["fb_group5"].isin(GROUP5)]
    frac_lost = m["lost"].mean()

    g = m.groupby("net_id")
    per = g.apply(lambda s: pd.Series({
        "n_lost": int(s["lost"].sum()), "n_ret": int(s["retained"].sum()),
        "med_lost": s.loc[s["lost"], metric].median(),
        "med_ret": s.loc[s["retained"], metric].median()})).reset_index()
    per = per.merge(gmap[["net_id", "fb_group5"]], on="net_id")
    paired = per[(per["n_lost"] >= MIN_PER_GROUP) & (per["n_ret"] >= MIN_PER_GROUP)].copy()
    paired["delta"] = paired["med_lost"] - paired["med_ret"]
    paired[["net_id", "fb_group5", "med_lost", "med_ret", "delta"]].to_csv(
        f"results/reproduce/nz_loss_{tag}_pernet{NZ_SUFFIX}.csv", index=False)

    rows = []
    for grp in ["ALL"] + GROUP5:
        sub = paired if grp == "ALL" else paired[paired["fb_group5"] == grp]
        if len(sub) < 10:
            continue
        w = stats.wilcoxon(sub["med_lost"], sub["med_ret"])       # two-sided
        rows.append({"group": grp, "n_networks": len(sub),
                     "median_delta": sub["delta"].median(),
                     "frac_delta_pos": (sub["delta"] > 0).mean(),
                     "wilcoxon_p": w.pvalue})
    pd.DataFrame(rows).to_csv(f"results/reproduce/nz_loss_{tag}_summary{NZ_SUFFIX}.csv", index=False)
    print(f"\n[Phase {'16' if metric=='stability_time' else '17'}] loss vs {metric}"
          f"  ({100*frac_lost:.0f}% of baseline-multimodal sets lost at noise {LOW})")
    for r in rows:
        print(f"  {r['group']:16s}: median delta(lost-ret)={r['median_delta']:+.3f}  "
              f"{100*r['frac_delta_pos']:.0f}% >0  (n={r['n_networks']}, p={r['wilcoxon_p']:.1e})")

loss_analysis("stability_time", "stability")
loss_analysis("cv_before_x1", "fluct")

# =============================================================================
# Phase 18 — robustness <-> pre-stability fluctuation by group (raw + partial)
# =============================================================================
runs = pd.read_csv(BASELINE, usecols=["net_id", "cv_before_x1"])
flu = runs.groupby("net_id")["cv_before_x1"].median().rename("flu_median").reset_index()
df18 = gmap.merge(flu, on="net_id", how="inner")
df18 = df18[(df18["fb_group5"].isin(GROUP5)) & (df18["multimodal_rate"] > 0)].dropna(subset=["flu_median"])
rows = []
for grp in GROUP5:
    sub = df18[df18["fb_group5"] == grp]
    r = stats.spearmanr(sub["multimodal_rate"], sub["flu_median"])
    lo, hi = spearman_ci(r.correlation, len(sub))
    pr, pp = partial_spearman(sub["multimodal_rate"].values, sub["flu_median"].values,
                              sub["n_edges"].values)
    rows.append({"fb_group5": grp, "n": len(sub), "spearman": r.correlation,
                 "ci_low": lo, "ci_high": hi, "p": r.pvalue,
                 "partial_density": pr, "partial_p": pp})
pd.DataFrame(rows).to_csv(f"results/reproduce/nz_robflu_correlations{NZ_SUFFIX}.csv", index=False)
print("\n[Phase 18] Spearman(robustness, median cv_before_x1) by group (raw | density-partial):")
for r in rows:
    print(f"  {r['fb_group5']:16s}: {r['spearman']:+.3f} [{r['ci_low']:+.2f},{r['ci_high']:+.2f}]"
          f"  | partial {r['partial_density']:+.3f} (p={r['partial_p']:.1e})")

# =============================================================================
# Phase 19 — within-network settling-time variability by group x noise
# =============================================================================
per_level = {}
for lvl in LEVELS:
    d = pd.read_csv(NOISE_FILES[lvl], usecols=["net_id", "stability_time"])
    g = d.groupby("net_id")["stability_time"]
    a = g.agg(n="count", mean="mean", std="std",
              q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75),
              med="median").reset_index()
    a["cv"] = a["std"] / a["mean"]
    a["rcv"] = (a["q3"] - a["q1"]) / a["med"]
    a = a[a["n"] >= MIN_RUNS].merge(gmap[["net_id", "fb_group5"]], on="net_id")
    per_level[lvl] = a[a["fb_group5"].isin(GROUP5)]

rows = []
for lvl in LEVELS:
    a = per_level[lvl]
    for grp in GROUP5:
        s = a[a["fb_group5"] == grp]
        rows.append({"noise": lvl, "fb_group5": grp, "n": len(s),
                     "median_cv": s["cv"].median(), "median_rcv": s["rcv"].median()})
pd.DataFrame(rows).to_csv(f"results/reproduce/nz_stvar_by_noise{NZ_SUFFIX}.csv", index=False)
# per-network values behind those medians, so the figure can show the distribution
# (one point per network) rather than only the group median
pd.concat([per_level[l].assign(noise=l)[["noise", "net_id", "fb_group5", "n", "cv", "rcv"]]
           for l in LEVELS], ignore_index=True).to_csv(
    f"results/reproduce/nz_stvar_per_network{NZ_SUFFIX}.csv", index=False)
print(f"\n[Phase 19] median rCV(settling time) by group across noise ({LVLTXT}):")
for grp in GROUP5:
    vals = [per_level[l].loc[per_level[l]["fb_group5"] == grp, "rcv"].median() for l in LEVELS]
    print(f"  {grp:16s}: {vals[0]:.3f} -> {vals[1]:.3f} -> {vals[2]:.3f}")

# =============================================================================
# Phase 15 — robustness & settling-time LEVELS across noise, by group
# =============================================================================
# Robustness = mean multimodal_rate over ALL group networks (rate = successful/1000,
# 0 if a network has no multimodal sets at that noise). Settling level = median over
# networks (with runs) of the per-network median stability_time.
allnets = feat[feat["fb_group5"].isin(GROUP5)][["net_id", "fb_group5"]]
lvl_rows = []
pernet_rows = []                         # per-network median settling (for the settling KDE ridge)
perrate_rows = []                        # per-network robustness rate (for the robustness KDE ridge)
for lvl in LEVELS:
    d = pd.read_csv(NOISE_FILES[lvl], usecols=["net_id", "stability_time"])
    cnt = d.groupby("net_id").size().rename("cnt")
    stmed = d.groupby("net_id")["stability_time"].median().rename("st_median")
    a = allnets.merge(cnt, on="net_id", how="left").merge(stmed, on="net_id", how="left")
    a["rate"] = a["cnt"].fillna(0) / 1000.0
    for grp in GROUP5:
        s = a[a["fb_group5"] == grp]
        lvl_rows.append({"noise": lvl, "fb_group5": grp, "n_networks": len(s),
                         "mean_rate": s["rate"].mean(),
                         "median_settling": s["st_median"].median()})
    pn = a.loc[a["st_median"].notna(), ["net_id", "fb_group5", "st_median"]].copy()
    pn.insert(0, "noise", lvl); pernet_rows.append(pn)
    pr = a[["net_id", "fb_group5", "rate"]].copy()           # all group networks (rate 0 incl.)
    pr.insert(0, "noise", lvl); perrate_rows.append(pr)
lvl_df = pd.DataFrame(lvl_rows)
lvl_df.to_csv(f"results/reproduce/nz_level_by_noise{NZ_SUFFIX}.csv", index=False)
pd.concat(pernet_rows, ignore_index=True).to_csv(
    f"results/reproduce/nz_settling_per_network{NZ_SUFFIX}.csv", index=False)
pd.concat(perrate_rows, ignore_index=True).to_csv(
    f"results/reproduce/nz_rate_per_network{NZ_SUFFIX}.csv", index=False)
print(f"\n[Phase 15] mean robustness (rate) by group across noise ({LVLTXT}):")
for grp in GROUP5:
    v = [lvl_df[(lvl_df.noise == l) & (lvl_df.fb_group5 == grp)]["mean_rate"].iloc[0] for l in LEVELS]
    print(f"  {grp:16s}: {v[0]:.3f} -> {v[1]:.3f} -> {v[2]:.3f}")

# =============================================================================
# Phase 15b — HR EXISTENCE (not robustness) by group x noise
# =============================================================================
# Existence = the network has >= 1 HR-enabling parameter set at that noise level,
# so the denominator is EVERY network in the category, not only those that ever
# produced HR. Reported for the same 2x2 tables (category vs all other categories
# at the SAME noise level, which nets out the overall noise effect):
#   rate          % of the category's networks that are HR-capable (Wilson 95% CI)
#   crude log-OR  unadjusted category contrast
#   MH log-OR     the same contrast DENSITY-CONTROLLED -- Mantel-Haenszel pooled
#                 over strata of n_edges, the project's standard adjustment, with
#                 the CMH test and a Breslow-Day homogeneity check. This matters
#                 here because the pure categories are sparse (median 5 edges) and
#                 the mixed ones dense (median 7), so the crude contrast partly
#                 measures density.
# Pool = non-disjoint (the canonical background for existence statistics, matching
# reproduce_hr_composition); disjoint networks never produce HR, so leaving them in
# the denominator would only dilute every category.
disj = {uid for uid, m in load_adjacency_matrices().items()
        if edge_bearing_components(m) >= 2}
pool = feat[feat["fb_group5"].isin(GROUP5) & ~feat["net_id"].isin(disj)][
    ["net_id", "fb_group5", "n_edges"]]

# Density-adjusted EXISTENCE RATE by g-computation -- the same model and estimator
# as reproduce_hr_composition, so the sigma=1.0 column of this table reproduces that
# figure's filled circles exactly. This is a different adjustment from the MH log-OR
# above: MH stratifies on n_edges and yields a contrast, g-computation models density
# smoothly and yields a RATE on the original percentage scale.
GC_DUM = GROUP5[1:]                      # reference = Pure Positive
GC_NBOOT = 500
GC_RNG = np.random.default_rng(0)


def _gc_design(sub):
    X = pd.DataFrame({"n_edges": sub.n_edges.astype(float),
                      "n_edges2": (sub.n_edges ** 2).astype(float)})
    for g in GC_DUM:
        X[g] = (sub.fb_group5 == g).astype(float)
    return sm.add_constant(X)


def _gc_rates(Xc_, y_):
    fit = sm.Logit(y_, Xc_).fit(disp=0)
    out = {}
    for g in GROUP5:
        Z = Xc_.copy()
        for gg in GC_DUM:
            Z[gg] = 1.0 if gg == g else 0.0
        out[g] = float(fit.predict(Z).mean())
    return out


def gcomp_existence(p):
    """Density-adjusted existence rate per category, with network-bootstrap 95% CIs."""
    Xc, y = _gc_design(p), p.hr.values.astype(float)
    point = _gc_rates(Xc, y)
    boot = {g: [] for g in GROUP5}
    for _ in range(GC_NBOOT):
        idx = GC_RNG.integers(0, len(p), len(p))
        try:
            r = _gc_rates(Xc.iloc[idx].reset_index(drop=True), y[idx])
        except Exception:
            continue
        for g in GROUP5:
            boot[g].append(r[g])
    ci = {g: np.percentile(boot[g], [2.5, 97.5]) for g in GROUP5}
    return point, ci


ex_rows = []
for lvl in LEVELS:
    hr_ids = set(pd.read_csv(NOISE_FILES[lvl], usecols=["net_id"])["net_id"].unique())
    p = pool.assign(hr=pool.net_id.isin(hr_ids).astype(int))
    adj_rate, adj_ci = gcomp_existence(p)
    for grp in GROUP5:
        p2 = p.assign(in_cat=(p.fb_group5 == grp).astype(int))
        n_in = int(p2.in_cat.sum())
        n_hr = int(p2.loc[p2.in_cat == 1, "hr"].sum())
        lo, hi = proportion_confint(n_hr, n_in, method="wilson")
        st = crude_and_mh(p2, "in_cat", outcome="hr", strata="n_edges")
        ex_rows.append({"noise": lvl, "fb_group5": grp, "n_networks": n_in,
                        "n_hr": n_hr, "rate": n_hr / n_in, "ci_low": lo, "ci_high": hi,
                        "crude_log_or": st["crude_log_or"],
                        "crude_ci_low": st["crude_ci_low"],
                        "crude_ci_high": st["crude_ci_high"],
                        "mh_log_or": st["mh_log_or"], "mh_ci_low": st["mh_ci_low"],
                        "mh_ci_high": st["mh_ci_high"], "cmh_p": st["cmh_p"],
                        "breslow_day_p": st["breslow_day_p"],
                        "n_strata": st["n_strata"], "mh_note": st["mh_note"],
                        "adj_rate": adj_rate[grp],
                        "adj_ci_low": adj_ci[grp][0], "adj_ci_high": adj_ci[grp][1]})
ex_df = pd.DataFrame(ex_rows)
ex_df.to_csv(f"results/reproduce/nz_existence_by_noise{NZ_SUFFIX}.csv", index=False)
print(f"\n[Phase 15b] HR EXISTENCE (% of category's networks) across noise ({LVLTXT}):")
for grp in GROUP5:
    s = ex_df[ex_df.fb_group5 == grp]
    rates = "  ".join(f"{r:.1%}" for r in s.rate)
    crude = "  ".join(f"{o:+.2f}" for o in s.crude_log_or)
    mh = "  ".join("  n/a" if not np.isfinite(o) else f"{o:+.2f}" for o in s.mh_log_or)
    print(f"  {grp:16s} n={s.n_networks.iloc[0]:4d}   rate {rates}")
    print(f"  {'':16s}   crude log-OR {crude}   density-controlled (MH) {mh}")
if ex_df.mh_note.astype(bool).any():
    for _, r in ex_df[ex_df.mh_note.astype(bool)].iterrows():
        print(f"    note: noise {r.noise} {r.fb_group5}: {r.mh_note}")

print(f"\n[set: {NOISE_SET}, levels {LVLTXT}]  "
      f"Saved -> results/reproduce/nz_*{NZ_SUFFIX}.csv")
