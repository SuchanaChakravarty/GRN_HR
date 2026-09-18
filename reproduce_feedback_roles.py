"""
Reproduction — distinct roles of 1/2/3-node positive (PFL) and negative (NFL)
feedback loops in multimodality, FIRST on IDENTITY, then on ROBUSTNESS (%).
==============================================================================
PART A — IDENTITY (can the network be multimodal at all?)
    Density-controlled (Mantel-Haenszel) presence log-OR + CMH, per category.
    Distinct roles: scale matters (crude 1-node < 2-node < 3-node); 1-node (self)
    vanishes under density control; 2-node retains a genuine density-independent
    effect; 3-node is not identifiable. Polarity barely matters (pos ~ neg).

PART B — ROBUSTNESS (over what fraction of parameter space, %)
    B1 (OVERALL): positive vs negative feedback have a SIMILAR overall influence on
        robustness — each additional loop of either polarity raises the per-set odds
        of multimodality by a comparable, modest amount, and the polarity gap is an
        order of magnitude smaller than the feedback-LENGTH effect. Shown BEFORE the
        per-length breakdown.
    B2 (BY LENGTH): binomial-GLM log-odds PER LOOP for the six elementary categories.
        Distinct roles: 1-node (self) loops REDUCE the rate; 2-node loops are the
        strongest positive driver; 3-node loops are positive. Polarity ~ symmetric
        within each scale (the small overall gap traces to the 1-node asymmetry).

Inputs : network_data.json + multimodal_results_combined_sorted.csv (raw only).
Depends: repro_common.py (shared) + motif_lib.py. Output: results/reproduce/
==============================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from repro_common import (build_network_table, crude_and_mh, binomial_glm_per_loop,
                          glm_contrast, ELEMENTARY, PRETTY, C_POS, C_NEG, savefig,
                          CONNECTED, OUT_SUFFIX)

warnings.filterwarnings("ignore")
os.makedirs("results/reproduce", exist_ok=True)

# scale-grouped display order (1-node, 2-node, 3-node; pos then neg within scale)
ORDER = ["fb_pos_1node", "fb_neg_1node", "fb_pos_2node",
         "fb_neg_2node", "fb_pos_3node", "fb_neg_3node"]


# =============================================================================
# 0. Build table from the two raw files
# =============================================================================
df = build_network_table(connected=CONNECTED)
print(f"{len(df)} networks ({df['label'].sum()} multimodal). "
      f"Six elementary feedback categories: {ELEMENTARY}\n")

# =============================================================================
# PART A — IDENTITY: density-controlled presence log-OR (Mantel-Haenszel + CMH)
# =============================================================================
print("=" * 78)
print("PART A — IDENTITY (binary multimodality): crude vs density-controlled log-OR")
print("=" * 78)
id_rows = [{"category": PRETTY[c], **crude_and_mh(df, "has_" + c)} for c in ORDER]
identity = pd.DataFrame(id_rows)
identity.to_csv(f"results/reproduce/feedback_roles_identity{OUT_SUFFIX}.csv", index=False)

show = identity.copy()
show["MH_logOR"] = show["mh_log_or"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "n/a")
show["MH_CI"] = show.apply(
    lambda r: f"[{r['mh_ci_low']:+.2f},{r['mh_ci_high']:+.2f}]"
    if pd.notna(r["mh_log_or"]) else "—", axis=1)
show["CMH_p"] = show["cmh_p"].map(lambda p: f"{p:.1e}" if pd.notna(p) else "n/a")
pd.set_option("display.width", 200, "display.max_columns", 25)
print(show[["category", "n_present", "pct_present", "pct_absent",
            "crude_log_or", "MH_logOR", "MH_CI", "CMH_p", "n_strata"]].to_string(index=False))

# =============================================================================
# PART B — ROBUSTNESS (multimodal rate, %)
# =============================================================================
print("\n" + "=" * 78)
print("PART B — ROBUSTNESS (multimodal rate, %): binomial-GLM log-odds PER LOOP")
print("=" * 78)

# ---------------------------------------------------------------------------
# B1. OVERALL — do positive and negative feedback have SIMILAR influence?
#     (shown BEFORE the per-length breakdown)
# ---------------------------------------------------------------------------
print("\n-- B1. OVERALL polarity: any positive vs any negative feedback --")
glm_agg = binomial_glm_per_loop(df, ["fb_pos_feedback", "fb_neg_feedback"])
agg = glm_agg.set_index("feature")
con = glm_contrast(glm_agg, "fb_pos_feedback", "fb_neg_feedback")

# scale-adjusted polarity effect: hold #loops per scale fixed, test net polarity
df_sa = df.assign(
    n1=df["fb_pos_1node"] + df["fb_neg_1node"],
    n2=df["fb_pos_2node"] + df["fb_neg_2node"],
    n3=df["fb_pos_3node"] + df["fb_neg_3node"],
    pol=df["fb_pos_feedback"] - df["fb_neg_feedback"])
glm_sa = binomial_glm_per_loop(df_sa, ["n1", "n2", "n3", "pol"]).set_index("feature")
pol = glm_sa.loc["pol"]
length_span = (glm_sa.loc[["n1", "n2", "n3"], "log_or"].min(),
               glm_sa.loc[["n1", "n2", "n3"], "log_or"].max())

for f, name in [("fb_pos_feedback", "Any positive feedback (PFL)"),
                ("fb_neg_feedback", "Any negative feedback (NFL)")]:
    r = agg.loc[f]
    print(f"  {name:30s}: log-OR/loop={r['log_or']:+.3f}  OR={r['or']:.3f}  "
          f"[{r['log_or_ci_low']:+.3f}, {r['log_or_ci_high']:+.3f}]")
print(f"  difference (PFL - NFL)        : {con['diff_log_or']:+.3f}  "
      f"[{con['ci_low']:+.3f}, {con['ci_high']:+.3f}]  Wald p={con['pval']:.2e}")
print(f"  scale-adjusted polarity effect : {pol['log_or']:+.3f}/loop "
      f"(net pos-minus-neg, length held fixed)")
print(f"  -> for reference, feedback-LENGTH effects span {length_span[0]:+.3f} to "
      f"{length_span[1]:+.3f}")
print("  => PFL and NFL have SIMILAR overall influence: both modestly raise robustness")
print("     (OR ~1.1-1.2, same direction); the polarity gap (~0.03-0.06 log-OR) is an")
print("     order of magnitude smaller than the feedback-length effect (~0.6 span).")
print("     (The small gap is resolvable only at this large n; it traces to the")
print("      1-node self-loop asymmetry visible in B2.)")

overall = pd.DataFrame([
    {"term": "Any positive feedback (PFL)", "log_or_per_loop": agg.loc["fb_pos_feedback", "log_or"],
     "or_per_loop": agg.loc["fb_pos_feedback", "or"],
     "ci_low": agg.loc["fb_pos_feedback", "log_or_ci_low"],
     "ci_high": agg.loc["fb_pos_feedback", "log_or_ci_high"]},
    {"term": "Any negative feedback (NFL)", "log_or_per_loop": agg.loc["fb_neg_feedback", "log_or"],
     "or_per_loop": agg.loc["fb_neg_feedback", "or"],
     "ci_low": agg.loc["fb_neg_feedback", "log_or_ci_low"],
     "ci_high": agg.loc["fb_neg_feedback", "log_or_ci_high"]},
    {"term": "difference (PFL - NFL)", "log_or_per_loop": con["diff_log_or"],
     "or_per_loop": np.nan, "ci_low": con["ci_low"], "ci_high": con["ci_high"],
     "wald_p": con["pval"]},
    {"term": "scale-adjusted polarity (net pos-neg)", "log_or_per_loop": pol["log_or"],
     "or_per_loop": pol["or"], "ci_low": pol["log_or_ci_low"], "ci_high": pol["log_or_ci_high"],
     "wald_p": pol["pval"]},
])
overall.to_csv(f"results/reproduce/feedback_roles_overall_polarity{OUT_SUFFIX}.csv", index=False)

# small figure: overall PFL vs NFL per-loop log-OR (similar)
fig, ax = plt.subplots(figsize=(7.5, 2.8))
for yp, (f, col) in zip([1, 0], [("fb_pos_feedback", C_POS), ("fb_neg_feedback", C_NEG)]):
    r = agg.loc[f]
    ax.plot([r["log_or_ci_low"], r["log_or_ci_high"]], [yp, yp], color=col, lw=3, zorder=1)
    ax.scatter(r["log_or"], yp, color=col, s=120, zorder=2, edgecolor="white", linewidth=1.2)
ax.axvline(0, color="black", lw=0.9, ls="--")
ax.set_yticks([1, 0]); ax.set_yticklabels(["Any positive\nfeedback (PFL)", "Any negative\nfeedback (NFL)"], fontsize=10)
ax.set_ylim(-0.6, 1.6)
ax.set_xlabel("log odds ratio per feedback loop (robustness)", fontsize=10)
ax.set_title("Overall: PFL and NFL have similar influence on robustness\n"
             f"(per-loop OR 1.19 vs 1.11; polarity gap {con['diff_log_or']:+.2f} "
             f"<< length span {length_span[1]-length_span[0]:.2f})",
             fontsize=10.5, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig(f"results/reproduce/feedback_roles_overall_polarity{OUT_SUFFIX}", bbox_inches="tight")

# ---------------------------------------------------------------------------
# B2. BY FEEDBACK LENGTH — six elementary categories, per-loop log-odds
# ---------------------------------------------------------------------------
print("\n-- B2. BY FEEDBACK LENGTH (1/2/3-node), per-loop log-odds --")
glm = binomial_glm_per_loop(df, ELEMENTARY)
glm_adj = binomial_glm_per_loop(df, ELEMENTARY + ["n_edges"])
rob_rows = []
for c in ORDER:
    g = glm[glm["feature"] == c].iloc[0]
    ga = glm_adj[glm_adj["feature"] == c].iloc[0]
    rob_rows.append({"category": PRETTY[c],
                     "logOR_per_loop": round(g["log_or"], 3), "OR_per_loop": round(g["or"], 3),
                     "ci_low": round(g["log_or_ci_low"], 3), "ci_high": round(g["log_or_ci_high"], 3),
                     "pval": g["pval"], "logOR_density_adj": round(ga["log_or"], 3)})
robustness = pd.DataFrame(rob_rows)
robustness.to_csv(f"results/reproduce/feedback_roles_robustness{OUT_SUFFIX}.csv", index=False)
print(f"(GLM dispersion = {glm.attrs['dispersion']:.1f} -> quasi-binomial CIs)")
print(robustness.to_string(index=False))

# =============================================================================
# Figure — two panels: identity (MH log-OR) | robustness by length (per-loop log-OR)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ypos = np.arange(len(ORDER))[::-1]
colcat = [C_POS if "pos" in c else C_NEG for c in ORDER]

ax = axes[0]
for yp, c, col in zip(ypos, ORDER, colcat):
    r = identity[identity["category"] == PRETTY[c]].iloc[0]
    ax.scatter(r["crude_log_or"], yp + 0.14, facecolor="white", edgecolor=col,
               s=80, zorder=3, linewidth=1.6)
    if pd.notna(r["mh_log_or"]):
        ax.plot([r["mh_ci_low"], r["mh_ci_high"]], [yp - 0.14, yp - 0.14], color=col, lw=2.4, zorder=1)
        ax.scatter(r["mh_log_or"], yp - 0.14, color=col, s=95, zorder=2, edgecolor="white", linewidth=1)
    else:
        ax.text(0.1, yp - 0.14, "MH n/a (only dense, all-multimodal strata)",
                va="center", fontsize=7.5, color="gray", style="italic")
ax.axvline(0, color="black", lw=0.9, ls="--")
ax.set_yticks(ypos); ax.set_yticklabels([PRETTY[c] for c in ORDER], fontsize=10)
ax.set_xlabel("log odds ratio for being multimodal\nopen = crude,  filled = density-controlled (MH, 95% CI)", fontsize=10)
ax.set_title("A. IDENTITY — can the network be multimodal?", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
for yp, c, col in zip(ypos, ORDER, colcat):
    r = robustness[robustness["category"] == PRETTY[c]].iloc[0]
    ax.plot([r["ci_low"], r["ci_high"]], [yp, yp], color=col, lw=2.4, zorder=1)
    ax.scatter(r["logOR_per_loop"], yp, color=col, s=95, zorder=2, edgecolor="white", linewidth=1)
ax.axvline(0, color="black", lw=0.9, ls="--")
ax.set_yticks(ypos); ax.set_yticklabels([PRETTY[c] for c in ORDER], fontsize=10)
ax.set_xlabel("log odds ratio PER additional loop\n(effect on per-parameter-set probability of multimodality)", fontsize=10)
ax.set_title("B2. ROBUSTNESS by feedback length", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

fig.legend(handles=[mpatches.Patch(color=C_POS, label="Positive feedback (PFL)"),
                    mpatches.Patch(color=C_NEG, label="Negative feedback (NFL)")],
           fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
plt.suptitle("Distinct Roles of 1/2/3-node Positive & Negative Feedback Loops in Multimodality",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 1])
savefig(f"results/reproduce/feedback_roles_forest{OUT_SUFFIX}", bbox_inches="tight")

# =============================================================================
# Figure (bar chart) — paper panel, grouped by scale + Overall, coloured by polarity
#   top   (Identity)   : crude log-OR (open bars) + density-controlled MH log-OR
#                        (filled bars), 95% CI error bars; "Overall" = PFL/NFL aggregate.
#   bottom (Robustness): binomial-GLM log-OR per loop (filled bars) + 95% CI;
#                        "Overall" = any-positive / any-negative feedback (from B1).
#   ≈0 (1-node identity), N/A (3-node identity), and off-scale crude (3-node) marked.
# =============================================================================
W = 0.34
C_PFBL, C_NFBL = "#C0625E", "#5E7FA6"            # subdued red (PFBL) / subdued blue (NFBL)
NA_FACE, NA_EDGE = "#E8E8E8", "#9E9E9E"
GROUPS = [("Overall",       "OVERALL_POS",  "OVERALL_NEG"),
          ("1-node (self)", "fb_pos_1node", "fb_neg_1node"),
          ("2-node",        "fb_pos_2node", "fb_neg_2node"),
          ("3-node",        "fb_pos_3node", "fb_neg_3node")]
GROUP_X = [0.0, 1.35, 2.35, 3.35]
DIVIDER = 0.675
LOW_TOP = 2.0                # identity broken axis: top of lower segment
HI_LO, HI_HI = 2.6, 8.9      # identity broken axis: upper segment (3-node crude + CI)

id_idx = identity.set_index("category")
id_ov = {"OVERALL_POS": crude_and_mh(df, "has_fb_pos_feedback"),
         "OVERALL_NEG": crude_and_mh(df, "has_fb_neg_feedback")}
rob_idx = robustness.set_index("category")
agg_idx = glm_agg.set_index("feature")


def id_data(key):
    d = id_ov[key] if key.startswith("OVERALL") else id_idx.loc[PRETTY[key]]
    return dict(crude=d["crude_log_or"], clo=d["crude_ci_low"], chi=d["crude_ci_high"],
                mh=d["mh_log_or"], mlo=d["mh_ci_low"], mhi=d["mh_ci_high"])


def rob_data(key):
    if key == "OVERALL_POS":
        r = agg_idx.loc["fb_pos_feedback"]
    elif key == "OVERALL_NEG":
        r = agg_idx.loc["fb_neg_feedback"]
    else:
        r = rob_idx.loc[PRETTY[key]]
        return dict(val=r["logOR_per_loop"], lo=r["ci_low"], hi=r["ci_high"])
    return dict(val=r["log_or"], lo=r["log_or_ci_low"], hi=r["log_or_ci_high"])


def ebar(ax, x, val, lo, hi, color):
    ax.errorbar(x, val, yerr=[[val - lo], [hi - val]], fmt="none", ecolor=color,
                elinewidth=1.4, capsize=3.5, capthick=1.4, zorder=6)


XLIM = (-0.58, 3.92)
YLAB_FS = 14


def draw_identity(ax, annotate):
    """Identity bars (crude open + adjusted filled + CIs) on one axis; text only
    when annotate=True (avoids duplicate labels across the broken sub-axes)."""
    for gx, (_, pkey, nkey) in zip(GROUP_X, GROUPS):
        for off, color, key in [(-W / 2, C_PFBL, pkey), (W / 2, C_NFBL, nkey)]:
            x = gx + off
            d = id_data(key)
            ax.bar(x, d["crude"], W, facecolor="none", edgecolor=color, lw=1.8, zorder=2)
            ebar(ax, x, d["crude"], d["clo"], d["chi"], color)
            if pd.isna(d["mh"]):
                h = 0.05
                ax.add_patch(mpatches.Rectangle((x - W / 2, -h), W, 2 * h, facecolor=NA_FACE,
                             edgecolor=NA_EDGE, hatch="xxxx", lw=1.0, zorder=4))
                if annotate:
                    ax.text(x, h + 0.05, "N/A", ha="center", va="bottom", fontsize=10.5,
                            color="#616161", style="italic", fontweight="bold")
            else:
                ax.bar(x, d["mh"], W, color=color, edgecolor="white", zorder=5)
                ebar(ax, x, d["mh"], d["mlo"], d["mhi"], "#2b2b2b")
                if annotate and abs(d["mh"]) < 0.1:
                    ax.text(x, d["mh"] - 0.05, "≈ 0", ha="center", va="top",
                            fontsize=10.5, fontweight="bold")
    ax.axhline(0, color="black", lw=1.0, zorder=1)
    ax.axvline(DIVIDER, color="#BBBBBB", lw=1.0, ls=":", zorder=0)


fig = plt.figure(figsize=(6.3, 9.2))
ax_rob = fig.add_axes([0.19, 0.07, 0.77, 0.38])   # bottom: Robustness
ax_lo  = fig.add_axes([0.19, 0.54, 0.77, 0.26])   # top: Identity, lower segment
ax_hi  = fig.add_axes([0.19, 0.815, 0.77, 0.10])  # top: Identity, broken upper segment

# --- TOP (Identity) — broken y-axis -----------------------------------------
draw_identity(ax_lo, annotate=True)
draw_identity(ax_hi, annotate=False)
# data-driven lower bound: just below the most-negative CI that sits in the lower segment
_idlo = min(v for _, pk, nk in GROUPS for key in (pk, nk) for d in [id_data(key)]
            for v in (d["clo"], d["mlo"]) if pd.notna(v) and v < LOW_TOP)
ax_lo.set_ylim(_idlo - 0.08, LOW_TOP); ax_lo.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
ax_hi.set_ylim(HI_LO, HI_HI);   ax_hi.set_yticks([4, 6, 8])
for a in (ax_lo, ax_hi):
    a.set_xlim(*XLIM); a.tick_params(axis="y", labelsize=13)
    a.tick_params(labelbottom=False, bottom=False)
    a.spines[["top", "right"]].set_visible(False)
ax_hi.spines["bottom"].set_visible(False)
ax_lo.spines["top"].set_visible(False)
# diagonal break marks
dk = dict(marker=[(-1, -1), (1, 1)], markersize=9, linestyle="none",
          color="k", mec="k", mew=1.2, clip_on=False)
ax_hi.plot([0, 1], [0, 0], transform=ax_hi.transAxes, **dk)
ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **dk)

# --- BOTTOM (Robustness) — per-loop GLM log-OR, filled ----------------------
for gx, (_, pkey, nkey) in zip(GROUP_X, GROUPS):
    for off, color, key in [(-W / 2, C_PFBL, pkey), (W / 2, C_NFBL, nkey)]:
        x = gx + off
        d = rob_data(key)
        ax_rob.bar(x, d["val"], W, color=color, edgecolor="white", zorder=3)
        ebar(ax_rob, x, d["val"], d["lo"], d["hi"], "#2b2b2b")
ax_rob.axhline(0, color="black", lw=1.0, zorder=1)
ax_rob.axvline(DIVIDER, color="#BBBBBB", lw=1.0, ls=":", zorder=0)
# data-driven y-limits: span all robustness bars + CIs with headroom
_rv = [v for _, pk, nk in GROUPS for key in (pk, nk) for d in [rob_data(key)] for v in (d["lo"], d["hi"])]
ax_rob.set_xlim(*XLIM); ax_rob.set_ylim(min(_rv) - 0.05, max(_rv) + 0.05)
ax_rob.tick_params(axis="y", labelsize=13)
ax_rob.set_xticks(GROUP_X)
ax_rob.set_xticklabels([g for g, _, _ in GROUPS], fontsize=14)
ax_rob.spines[["top", "right"]].set_visible(False)

# --- y-axis labels (same font size, centred over each panel region) ---------
fig.text(0.05, (0.54 + 0.915) / 2, "Contingency-table log-OR\n(HR Existence)",
         rotation=90, va="center", ha="center", fontsize=YLAB_FS)
fig.text(0.05, (0.07 + 0.45) / 2, "Binomial-GLM log-OR per loop\n(HR Robustness)",
         rotation=90, va="center", ha="center", fontsize=YLAB_FS)

# --- legend (above the broken top panel) ------------------------------------
fig.legend(handles=[
    mpatches.Patch(color=C_PFBL, label="PFBL"),
    mpatches.Patch(color=C_NFBL, label="NFBL"),
    mpatches.Patch(facecolor="none", edgecolor="#555", lw=1.6, label="crude (open)"),
    mpatches.Patch(facecolor="#888", edgecolor="white", label="adjusted (filled)"),
    mpatches.Patch(facecolor=NA_FACE, edgecolor=NA_EDGE, hatch="xxxx",
                   label="N/A (100% HR)")],
    loc="upper center", bbox_to_anchor=(0.57, 1.005), ncol=3, frameon=False,
    fontsize=11.5, columnspacing=1.4, handletextpad=0.6)
savefig(f"results/reproduce/feedback_roles_barchart{OUT_SUFFIX}", bbox_inches="tight")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 78)
print("DISTINCT ROLES SUMMARY")
print("=" * 78)
print("OVERALL polarity (robustness): PFL ~ NFL — both modestly raise robustness "
      "(OR 1.19 vs 1.11); gap << feedback-length effect.")
print("IDENTITY (density-controlled MH log-OR): 1-node vanishes, 2-node retained, "
      "3-node not identifiable; pos ~ neg.")
print("ROBUSTNESS by length (per-loop log-OR): 1-node (self) REDUCES rate, 2-node "
      "strongest positive, 3-node positive; pos ~ neg within scale.")
print(f"\n[pool: {'connected-only' if CONNECTED else 'full'}]")
print(f"Saved: results/reproduce/feedback_roles_{{overall_polarity,identity,robustness}}{OUT_SUFFIX}.csv")
print(f"       results/reproduce/feedback_roles_{{overall_polarity,forest,barchart}}{OUT_SUFFIX}.{{png,svg}}")
