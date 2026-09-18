"""
reproduce_stability_dynamics.py
==============================================================================
Standalone reproduction of the KEY CONCLUSIONS of Phases 8-10 — the DYNAMICS of
multimodality (how fast a circuit settles, and how much it fluctuates on the way)
as opposed to its capacity/robustness (Phases 3-7). Depends only on the two raw
files via repro_common (build_dynamics_table / build_feature_table).

Unifying result: capacity & robustness are polarity-blind and scale-driven, but
DYNAMICS ARE POLARITY-DRIVEN.

Phase 8  (settling time, `stability_time`)
  * Predictive ceiling: GBM 5-fold R^2 on per-network median settling time.
  * OLS time-units PER feedback loop (crude -> density-adjusted, HC3 SE):
    every NEGATIVE loop shortens settling, every POSITIVE loop lengthens it.
Phase 9  (settling time vs robustness)
  * Network-level Spearman(multimodal_rate, median stability_time): weak overall,
    and the SIGN FLIPS across connectivity classes (Simpson's-flavour).
Phase 10 (pre-stability fluctuation, `cv_before_x1`)
  * Fluctuation median by category: rises with connectivity and POSITIVE feedback.
  * Robustness<->fluctuation network-wise Spearman: weak +, concentrated in
    negative-dominant / sparse circuits.
  * Within-network Spearman(stability_time, cv_before_x1): consistently positive.

Feedback is grouped 5 ways (fb_group5: Pure Positive / Mixed Pos-Dom / Balanced /
Mixed Neg-Dom / Pure Negative); settling time and fluctuation are matched analyses.

The SECOND metric (everything paired with settling time) is pluggable via the
DYN_METRIC2 env var -- see repro_common.METRIC2:
    fluct      (default) median pre-settling fluctuation, cv_before_x1
    overshoot            fraction of the network's HR sets that overshoot, from
                         frac_data/; within_corr becomes the within-network
                         Spearman(stability_time, overshoot)
The overshoot run tags every output "_overshoot" ahead of OUT_SUFFIX, so the two
metrics' results coexist:
    DYN_METRIC2=overshoot HR_RESULTS=summary_result_noise1_frac10.csv \\
        python reproduce_stability_dynamics.py

Outputs (results/reproduce/, NO figures -- see visualize_stability_dynamics.py).
Names below are the default-metric ones; the overshoot run substitutes
dyn_overshoot_ols / dyn_overshoot_by_category / dyn_robust_overshoot_strat and
appends "_overshoot" to the rest:
    dyn_per_network.csv                per-network dynamics + category table (incl. fb_group5)
    dyn_stability_ols.csv              per-loop OLS coefs on settling time (crude + density-adj)
    dyn_fluct_ols.csv                  per-loop OLS coefs on metric 2 (crude + density-adj)
    dyn_stability_predictive.csv       Phase 8: 5-fold CV R^2 (baseline / RF / GBM)
    dyn_rate_timing_strat.csv          Phase 9: overall + stratified rate<->timing Spearman
    dyn_fluct_by_category.csv          Phase 10(1): median metric 2 by category
    dyn_robust_fluct_strat.csv         Phase 10(2): overall + stratified robustness<->metric 2
    dyn_within_corr_summary.csv        Phase 10(3): within-network stability<->metric 2
    dyn_gradient_stats.csv             5-group gradient: monotone-trend (Spearman/Kendall) +
                                       adjacent-neighbor Cliff's delta and Holm-corrected MWU p

Runtime: a couple of minutes (dominated by the Phase 8 GBM 5-fold CV). Run from root:
    python reproduce_stability_dynamics.py
==============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score

from repro_common import (build_feature_table, build_dynamics_table, spearman_ci,
                          ELEMENTARY, PRETTY, GROUP5, OUT_SUFFIX,
                          METRIC2, M2_TAG, M2_COL, DYN_METRIC2, overshoot_per_network,
                          path_per_network)
from motif_lib import DEPRECATED_FEATURE_COLS

warnings.filterwarnings("ignore")
os.makedirs("results/reproduce", exist_ok=True)
RNG = 42

# category systems reused by Phases 9 & 10 (feedback = the adopted 5-way grouping)
GROUP5 = ["Pure Positive", "Mixed Pos-Dom", "Balanced", "Mixed Neg-Dom", "Pure Negative"]
SYSTEMS = [
    ("SCC profile", "scc_profile", ["(3) strongly conn.", "(2,1)", "(1,1,1)"]),
    ("Density",     "density_bin", ["<=4 edges", "5 edges", "6 edges", ">=7 edges"]),
    ("Feedback",    "fb_group5",   GROUP5),
]

# =============================================================================
# 0. Build tables from the two raw files
# =============================================================================
feat = build_feature_table()                         # full topology (also feeds Phase 8 model)
net = build_dynamics_table(feature_df=feat)          # per-network dynamics + categories

# Second metric (bottom panel of every dynamics figure) is pluggable -- see
# repro_common.METRIC2. For "overshoot" it replaces the fluctuation column with the
# fraction of each network's HR sets that overshoot, and within_corr with the
# within-network Spearman(stability_time, overshoot).
if DYN_METRIC2 == "overshoot":
    ov = overshoot_per_network()
    net = net.merge(ov[["net_id", "over_rate", "over_corr"]], on="net_id", how="left")
    net["within_corr"] = net["over_corr"]
    print(f"  metric 2 = overshoot fraction ({net.over_rate.notna().sum()} networks matched)")
elif DYN_METRIC2 in ("path", "pathraw", "traversal"):
    pl = path_per_network(normalize=(DYN_METRIC2 != "pathraw"),
                          per_time=(DYN_METRIC2 == "traversal"))
    net = net.merge(pl[["net_id", "path_median", "path_corr"]], on="net_id", how="left")
    net["within_corr"] = net["path_corr"]
    print(f'  metric 2 = {METRIC2["desc"]} ({net.path_median.notna().sum()} networks matched)')

net.to_csv(f"results/reproduce/dyn_per_network{M2_TAG}{OUT_SUFFIX}.csv", index=False)
print(f"{len(net)} multimodal networks | "
      f"categories: {net['category'].value_counts().to_dict()}")

# =============================================================================
# 1. Phase 8 — settling time: predictive ceiling + per-loop OLS
# =============================================================================
META = ["net_id", "label", "multimodal_count", "multimodal_rate",
        "label_t05", "label_t10", "label_t25"]
FEAT_COLS = [c for c in feat.columns              # Phase 8 set: all topology minus deprecated
             if c not in META and c not in DEPRECATED_FEATURE_COLS]
model_df = net[["net_id", "st_median", M2_COL, "n_runs"]].merge(feat, on="net_id", how="left")
X, y = model_df[FEAT_COLS].values, model_df["st_median"].values
cv = KFold(5, shuffle=True, random_state=RNG)

def cv_r2(model):
    p = np.clip(cross_val_predict(model, X, y, cv=cv, n_jobs=-1), 0, None)
    return r2_score(y, p), stats.spearmanr(y, p).correlation

pred_rows = []
for name, m in [("Mean baseline", DummyRegressor(strategy="mean")),
                ("Random Forest", RandomForestRegressor(400, n_jobs=-1, random_state=RNG)),
                ("Gradient Boosting", GradientBoostingRegressor(
                    n_estimators=400, learning_rate=0.05, max_depth=3,
                    subsample=0.8, random_state=RNG))]:
    r2, sp = cv_r2(m)
    pred_rows.append({"model": name, "cv_r2": r2, "spearman": sp})
    print(f"  [Phase 8] {name:18s}: R2={r2:.3f}  Spearman={sp:.3f}")
pd.DataFrame(pred_rows).to_csv(f"results/reproduce/dyn_stability_predictive{M2_TAG}{OUT_SUFFIX}.csv", index=False)

# OLS: change in per-network median TARGET per feedback loop (crude -> +n_edges), HC3 SE.
# Fit for BOTH dynamic targets so settling time and fluctuation are matched analyses.
def fit_ols(target, cols, label):
    sub = model_df[[target] + cols].dropna()
    fit = sm.OLS(sub[target].astype(float).values,
                 sm.add_constant(sub[cols].astype(float))).fit(cov_type="HC3")
    ci = fit.conf_int()
    return pd.DataFrame({"feature": ["const"] + list(cols), "model": label,
                         "coef": fit.params.values,
                         "ci_low": ci[0].values, "ci_high": ci[1].values,
                         "pval": fit.pvalues.values})

def ols_table(target):
    t = pd.concat([fit_ols(target, ELEMENTARY, "crude"),
                   fit_ols(target, ELEMENTARY + ["n_edges"], "density_adjusted")],
                  ignore_index=True)
    t = t[t["feature"] != "const"].reset_index(drop=True)
    t["pretty"] = t["feature"].map(PRETTY).fillna(t["feature"])
    return t

for target, tag, out in [("st_median", "median stability_time", f"dyn_stability_ols{M2_TAG}{OUT_SUFFIX}.csv"),
                         (M2_COL, METRIC2["desc"], f'{METRIC2["ols"]}{OUT_SUFFIX}.csv')]:
    tbl = ols_table(target)
    tbl.to_csv(f"results/reproduce/{out}", index=False)
    print(f"\n[Phase 8/10] per-loop OLS on {tag} (density-adjusted):")
    adj = tbl[tbl.model == "density_adjusted"]
    for _, r in adj[adj.feature.isin(ELEMENTARY)].iterrows():
        print(f"  {r['pretty']:20s}: {r['coef']:+8.3f} per loop  (p={r['pval']:.1e})")

# =============================================================================
# 2. Phase 9 — settling time vs robustness: overall + stratified (sign flip)
# =============================================================================
def strat_spearman(xcol, ycol):
    rows = []
    r = stats.spearmanr(net[xcol], net[ycol], nan_policy="omit").correlation
    lo, hi = spearman_ci(r, net[[xcol, ycol]].dropna().shape[0])
    rows.append({"system": "Overall", "category": "all", "n": len(net.dropna(subset=[xcol, ycol])),
                 "spearman": r, "ci_low": lo, "ci_high": hi})
    for sysname, col, levels in SYSTEMS:
        for lvl in levels:
            s = net[net[col] == lvl].dropna(subset=[xcol, ycol])
            if len(s) < 10:
                continue
            rr = stats.spearmanr(s[xcol], s[ycol]).correlation
            l, h = spearman_ci(rr, len(s))
            rows.append({"system": sysname, "category": lvl, "n": len(s),
                         "spearman": rr, "ci_low": l, "ci_high": h})
    return pd.DataFrame(rows)

rate_timing = strat_spearman("multimodal_rate", "st_median")
rate_timing.to_csv(f"results/reproduce/dyn_rate_timing_strat{M2_TAG}{OUT_SUFFIX}.csv", index=False)
ov = rate_timing.iloc[0]
print(f"\n[Phase 9] Spearman(rate, median stability_time): overall {ov['spearman']:+.3f}; "
      f"stratified range [{rate_timing['spearman'][1:].min():+.3f}, "
      f"{rate_timing['spearman'][1:].max():+.3f}] (sign flips across connectivity)")

# =============================================================================
# 3. Phase 10 — fluctuations: by category, robustness link, within-network
# =============================================================================
# (1) median cv_before_x1 by category
fl_rows = []
for sysname, col, levels in SYSTEMS + [
        ("Robustness quartile", "rate_q",
         ["Q1 (least robust)", "Q2", "Q3", "Q4 (most robust)"])]:
    vals = [net.loc[net[col] == lvl, M2_COL].dropna() for lvl in levels]
    kw = stats.kruskal(*[v for v in vals if len(v)])
    for lvl, v in zip(levels, vals):
        fl_rows.append({"system": sysname, "category": lvl, "n": len(v),
                        METRIC2["bycat_col"]: v.median(), "kw_p": kw.pvalue})
pd.DataFrame(fl_rows).to_csv(f'results/reproduce/{METRIC2["bycat"]}{OUT_SUFFIX}.csv', index=False)

# (2) robustness <-> fluctuation, network-wise (overall + stratified)
robust_fluct = strat_spearman("multimodal_rate", M2_COL)
robust_fluct.to_csv(f'results/reproduce/{METRIC2["strat"]}{OUT_SUFFIX}.csv', index=False)
print(f'[Phase 10] Spearman(rate, {METRIC2["desc"]}): overall '
      f"{robust_fluct.iloc[0]['spearman']:+.3f}; stratified range "
      f"[{robust_fluct['spearman'][1:].min():+.3f}, {robust_fluct['spearman'][1:].max():+.3f}]")

# (3) within-network stability_time <-> fluctuation
wc = net["within_corr"].dropna()
wilcox_p = stats.wilcoxon(wc).pvalue
pd.DataFrame([{"n_networks": len(wc), "within_corr_mean": wc.mean(),
               "within_corr_median": wc.median(), "frac_positive": (wc > 0).mean(),
               "wilcoxon_p": wilcox_p}]).to_csv(
    f"results/reproduce/dyn_within_corr_summary{M2_TAG}{OUT_SUFFIX}.csv", index=False)
print(f'[Phase 10] {METRIC2["within"]}: '
      f"median {wc.median():+.3f}, {100*(wc>0).mean():.0f}% positive "
      f"({len(wc)} nets, Wilcoxon p={wilcox_p:.1e})")

# =============================================================================
# 4. Gradient statistics across the ORDERED 5 feedback groups (dyn_dist_violin)
# =============================================================================
# Monotone-trend test (Spearman/Kendall on the ordinal group code) + adjacent-
# neighbor effect sizes (Cliff's delta) with Holm-corrected Mann-Whitney p. Cliff's
# delta is the headline (n is large -> p is tiny for almost any real difference).
CODE = {g: i for i, g in enumerate(GROUP5)}
def cliffs_delta(a, b):
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1
def holm(pv):
    order = np.argsort(pv); adj = np.empty(len(pv)); prev = 0.0; mtot = len(pv)
    for rank, idx in enumerate(order):
        prev = max(prev, min(1.0, (mtot - rank) * pv[idx])); adj[idx] = prev
    return adj

grad_rows = []
for metric, mlabel in [("st_median", "settling_time"), (M2_COL, METRIC2["grad"])]:
    d = net.dropna(subset=[metric])
    d = d[d["fb_group5"].isin(GROUP5)]
    x = d["fb_group5"].map(CODE).values
    rho, rp = stats.spearmanr(x, d[metric].values)
    tau, tp = stats.kendalltau(x, d[metric].values)
    grad_rows.append({"metric": mlabel, "comparison": "trend(all groups)",
                      "spearman_rho": rho, "spearman_p": rp, "kendall_tau": tau,
                      "kendall_p": tp, "cliffs_delta": np.nan, "mwu_p_holm": np.nan})
    pairs = [(GROUP5[i], GROUP5[i + 1]) for i in range(4)]
    praw = [stats.mannwhitneyu(d.loc[d.fb_group5 == a, metric],
                               d.loc[d.fb_group5 == b, metric],
                               alternative="two-sided").pvalue for a, b in pairs]
    deltas = [cliffs_delta(d.loc[d.fb_group5 == a, metric].values,
                           d.loc[d.fb_group5 == b, metric].values) for a, b in pairs]
    for (a, b), dl, pa in zip(pairs, deltas, holm(praw)):
        grad_rows.append({"metric": mlabel, "comparison": f"{a} -> {b}",
                          "spearman_rho": np.nan, "spearman_p": np.nan,
                          "kendall_tau": np.nan, "kendall_p": np.nan,
                          "cliffs_delta": dl, "mwu_p_holm": pa})
pd.DataFrame(grad_rows).to_csv(f"results/reproduce/dyn_gradient_stats{M2_TAG}{OUT_SUFFIX}.csv", index=False)
print("\n[Gradient] monotone trend (Spearman rho on ordered groups) + neighbor Cliff's delta:")
for r in grad_rows:
    if r["comparison"] == "trend(all groups)":
        print(f"  {r['metric']:13s} trend rho={r['spearman_rho']:+.3f} (p={r['spearman_p']:.1e})")
    else:
        print(f"    {r['comparison']:32s}: Cliff's d={r['cliffs_delta']:+.3f}  (Holm p={r['mwu_p_holm']:.1e})")

print("\nSaved -> results/reproduce/dyn_*.csv")
