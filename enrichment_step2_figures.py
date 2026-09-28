"""
STANDALONE reproduction pipeline for enrichment analysis (2/2) — from gene_table.csv, compute
the exact numbers behind the two figures and render them.

Figures (written to reproduce/):
  fig_enrichment_and_length.{png,svg}
      A: enrichment (any/pos/neg feedback) of feedback-rich genes for 4 functional
         sets, at TWO thresholds -- outline bars z>0, filled bars z>=2 (the strict
         null-model, degree-controlled definition). 'Any' Fisher p-values annotated
         (bold p for z>=2, regular p for z>0).
      B: functional enrichment vs feedback-loop length (Mantel-Haenszel, degree-
         controlled), 1/2/3-node; differentiation's 3-node point starred.
  venn_feedback_thresholds.{png,svg}
      Two 4-set Venns (feedback z>0 and z>=2) vs the three functional sets, within
      the annotated-regulator background.

All statistics recomputed here from gene_table.csv (no external result files).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import to_rgba
import venn

ROOT = Path(__file__).resolve().parent
CATS = ["stem_progenitor", "cellular_development", "differentiation", "homeostasis"]
COL = {"stem_progenitor": "#6a51a3", "cellular_development": "#e08214",
       "differentiation": "#2c7fb8", "fb": "#41ab5d"}


# ------------------------------------------------------------------- stats #
def crude_logor(a, b, c, d):
    if min(a, b, c, d) == 0:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    return np.log((a * d) / (b * c)), np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)


def contingency(df, feat, cat):
    f = df[feat].values.astype(bool); c = df[cat].values.astype(bool)
    return int((f & c).sum()), int((f & ~c).sum()), int((~f & c).sum()), int((~f & ~c).sum())


def mh_logor(df, feat, cat, stratum):
    R = S = PR = PSQR = QS = 0.0
    for _, sub in df.groupby(stratum):
        a, b, c, d = contingency(sub, feat, cat)
        n = a + b + c + d
        if n == 0:
            continue
        Rk, Sk = a * d / n, b * c / n
        P, Q = (a + d) / n, (b + c) / n
        R += Rk; S += Sk; PR += P * Rk; PSQR += P * Sk + Q * Rk; QS += Q * Sk
    if R == 0 or S == 0:
        return np.nan, np.nan
    return np.log(R / S), np.sqrt(PR / (2 * R**2) + PSQR / (2 * R * S) + QS / (2 * S**2))


def enrich(bg, mask, cat):
    a, b, c, d = contingency(bg.assign(_f=mask), "_f", cat)
    lor, se = crude_logor(a, b, c, d)
    _, p = fisher_exact([[a, b], [c, d]])
    return lor, se, p


def pfmt(p):
    if p >= 1e-3:
        return f"{p:.3f}"
    mant, exp = f"{p:.0e}".split("e")
    return f"{mant}×10$^{{{int(exp)}}}$"


def compute(bg):
    """Enrichment at BOTH feedback thresholds (z>=2 filled, z>0 outline) + length gradient."""
    bg = bg.copy()
    bg["deg_stratum"] = pd.qcut(bg["degree"].rank(method="first"), 5, labels=False)
    enr = {}   # cat -> sign -> {z2:(lor,se,p), z0:(lor,se,p)}
    zcol = {"any": "z", "pos": "z_pos", "neg": "z_neg"}
    for cat in CATS:
        enr[cat] = {sign: {"z2": enrich(bg, bg[c] >= 2, cat), "z0": enrich(bg, bg[c] > 0, cat)}
                    for sign, c in zcol.items()}
    grad = {cat: {k: mh_logor(bg, f"sig_scale{k}", cat, "deg_stratum") for k in (1, 2, 3)}
            for cat in CATS}
    return enr, grad


# ------------------------------------------------------------- figure 1 #
def figure_enrichment_and_length(bg):
    enr, _ = compute(bg)
    plt.rcParams.update({"font.size": 13})
    fig = plt.figure(figsize=(4.165, 8.69))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.54, wspace=0.1)
    axB = fig.add_subplot(gs[0, :])
    axGL = fig.add_subplot(gs[1, 0]); axGR = fig.add_subplot(gs[1, 1], sharey=axGL)
    green = "#41ab5d"
    SIGN_COL = {"any": green, "pos": "#2c7fb8", "neg": "#d95f0e"}

    # ---------- Panel A: enrichment bars, two thresholds (outline z>0, filled z>=2) ----------
    groups = [("stem_progenitor", "Stemness"), ("cellular_development", "Cellular Dev."),
              ("differentiation", "Cell Diff."), ("homeostasis", "Homeostasis")]
    x = np.arange(len(groups)); bw = 0.135
    subbars = [("any", "z0", False), ("any", "z2", True),
               ("pos", "z0", False), ("pos", "z2", True),
               ("neg", "z0", False), ("neg", "z2", True)]
    for k, (sign, thr, filled) in enumerate(subbars):
        off = (k - 2.5) * bw; col = SIGN_COL[sign]
        vals = [enr[cat][sign][thr][0] for cat, _ in groups]
        errs = [1.96 * enr[cat][sign][thr][1] for cat, _ in groups]
        if filled:
            axB.bar(x + off, vals, bw, yerr=errs, capsize=1.8, color=col, zorder=2)
        else:
            axB.bar(x + off, vals, bw, yerr=errs, capsize=1.8, facecolor="none",
                    edgecolor=col, lw=1.4, zorder=2)
    axB.axhline(0, color="k", lw=0.8, ls="--")
    axB.set_ylim(-0.95, 1.9)
    axB.spines[["top", "right"]].set_visible(False)
    for i, (cat, _) in enumerate(groups):
        px = x[i] - 0.8 * bw
        for thr, y, wt in (("z2", 1.30, "bold"), ("z0", 1.12, "normal")):
            p = enr[cat]["any"][thr][2]
            t1 = TextArea("p=", textprops=dict(style="italic", weight=wt, color=green, fontsize=8))
            t2 = TextArea(pfmt(p), textprops=dict(color="black", fontsize=8))
            box = HPacker(children=[t1, t2], align="baseline", pad=0, sep=0)
            axB.add_artist(AnnotationBbox(box, (px, y), frameon=False, box_alignment=(0.5, 0.0)))
    axB.set_xticks(x); axB.set_xticklabels([g[1] for g in groups], fontsize=11, rotation=18, ha="right")
    axB.tick_params(axis="y", labelsize=12)
    axB.set_ylabel("Log-OR (Feedback Vs Rest)", fontsize=12)
    sign_leg = [Patch(color=SIGN_COL["any"], label="Any Feedback"),
                Patch(color=SIGN_COL["pos"], label="Positive"),
                Patch(color=SIGN_COL["neg"], label="Negative")]
    leg1 = axB.legend(handles=sign_leg, fontsize=10, loc="lower left", frameon=False)
    axB.add_artist(leg1)
    fill_leg = [Patch(facecolor="#888", edgecolor="k", label="$z$ ≥ 2  (strict; bold $p$)"),
                Patch(facecolor="none", edgecolor="#888", label="$z$ > 0  (permissive; reg. $p$)")]
    axB.legend(handles=fill_leg, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.17),
               frameon=False, title="Null-Model $z$ (Degree-Controlled)", title_fontsize=8.5)

    # ---------- Panel B: two-axes per-scale length gradient (permissive | stringent) ----------
    GRAD = [("stem_progenitor", "Stemness", "#6a51a3", "-"),
            ("cellular_development", "Cellular Dev.", "#e08214", "-"),
            ("differentiation", "Cell Diff.", "#2c7fb8", "-")]

    def gfeat(scale, cutoff):
        if scale == 1:
            return bg["sig_scale1"]
        zc = bg["z2"] if scale == 2 else bg["z3"]
        return zc > 0 if cutoff == "perm" else zc >= 2
    for ax, cutoff, ttl in ((axGL, "perm", "Permissive ($z_k$>0)"),
                            (axGR, "strin", "Stringent ($z_k$≥2)")):
        ax.axhline(0, color="#888", lw=0.9, ls=":")
        for cat, label, col, ls in GRAD:
            pts = [enrich(bg, gfeat(k, cutoff), cat) for k in (1, 2, 3)]
            ys = [p[0] for p in pts]; es = [1.96 * p[1] for p in pts]
            ax.errorbar([1, 2, 3], ys, yerr=es, marker="o", ms=6, lw=2.0, ls=ls, color=col,
                        capsize=3.6, elinewidth=1.6, markeredgecolor="white",
                        markeredgewidth=0.6, label=label, zorder=3)
        ax.set_xticks([1, 2, 3]); ax.set_xlim(0.7, 3.3)
        ax.set_title(ttl, fontsize=10)
        ax.grid(axis="y", ls=":", color="#dddddd", lw=0.8); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axGL.set_ylim(-0.35, 1.5)
    axGL.set_ylabel("Log-OR (Scale-$k$ Feedback-Rich\nVs Rest, Degree-Controlled Null)", fontsize=9)
    axGL.tick_params(labelsize=10); axGR.tick_params(labelleft=False)
    axGR.legend(fontsize=7.8, loc="upper right", frameon=False)
    fig.text(0.57, 0.055, "Feedback-Loop Length (Nodes)", ha="center", fontsize=11)

    for ext in ("png", "svg"):
        fig.savefig(ROOT / f"fig_enrichment_and_length.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    for cat in CATS:
        a2 = enr[cat]["any"]["z2"]; a0 = enr[cat]["any"]["z0"]
        print(f"  {cat:22s} z>=2 {a2[0]:+.2f} (p={a2[2]:.3g}) | z>0 {a0[0]:+.2f} (p={a0[2]:.3g})")


# ------------------------------------------------------------- figure 2 #
def figure_venn(bg):
    S = set(bg.loc[bg["stem_progenitor"], "gene"])
    C = set(bg.loc[bg["cellular_development"], "gene"])
    D = set(bg.loc[bg["differentiation"], "gene"])
    F0 = set(bg.loc[bg["z"] > 0, "gene"])
    F2 = set(bg.loc[bg["z"] >= 2, "gene"])
    N = len(bg); FS = 14
    fcol = {k: to_rgba(COL[k], 0.45) for k in ("stem_progenitor", "cellular_development", "differentiation", "fb")}

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
    for ax, F, fbname in [(axes[0], F0, f"Feedback: $z$ > 0 (Permissive)  (N = {len(F0)})"),
                          (axes[1], F2, f"Feedback-Rich: $z$ ≥ 2 (Strict)  (N = {len(F2)})")]:
        data = {"S": S, "C": C, "D": D, "F": F}
        petals = venn.generate_petal_labels(data.values(), fmt="{size}")
        venn.draw_venn(petal_labels=petals, dataset_labels=list(data.keys()), hint_hidden=False,
                       colors=[fcol["stem_progenitor"], fcol["cellular_development"],
                               fcol["differentiation"], fcol["fb"]],
                       figsize=(8, 8), fontsize=FS, legend_loc="upper left", ax=ax)
        if ax.get_legend():
            ax.get_legend().remove()
        ylo, yhi = ax.get_ylim(); ax.set_ylim(ylo, ylo + 0.86 * (yhi - ylo))
        ax.legend(handles=[Patch(facecolor=fcol["fb"], edgecolor="k", label=fbname)],
                  loc="upper left", fontsize=FS, frameon=False)
        ax.add_patch(Rectangle((0.012, 0.012), 0.976, 0.976, transform=ax.transAxes,
                               fill=False, ec="#555", lw=1.4, zorder=0))
        none = N - len(S | C | D | F)
        ax.text(0.5, 0.02, f"Background: Annotated Regulators = {N:,}\n"
                f"(in None of the Four Sets: {none:,})", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=FS - 2, style="italic", color="#555")

    shared = [Patch(facecolor=fcol["stem_progenitor"], edgecolor="k", label=f"Stemness (Combined)  (N = {len(S)})"),
              Patch(facecolor=fcol["cellular_development"], edgecolor="k", label=f"Cellular Development  (N = {len(C)})"),
              Patch(facecolor=fcol["differentiation"], edgecolor="k", label=f"Cell Differentiation  (N = {len(D)})")]
    fig.legend(handles=shared, loc="upper center", ncol=3, fontsize=FS, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "svg"):
        fig.savefig(ROOT / f"venn_feedback_thresholds.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  |S|={len(S)} |C|={len(C)} |D|={len(D)} |F0|={len(F0)} |F2|={len(F2)} bg={N}")


def main():
    df = pd.read_csv(ROOT / "gene_table.csv", keep_default_na=False)
    for c in ("z", "z_pos", "z_neg", "z2", "z3", "degree", "out_degree", "n_go_bp"):
        df[c] = pd.to_numeric(df[c])
    for c in CATS + ["sig_scale1", "sig_scale2", "sig_scale3"]:
        df[c] = df[c].astype(str).isin(["True", "true", "1"])
    bg = df[(df["out_degree"] > 0) & (df["n_go_bp"] > 0)].copy()
    print(f"background (annotated regulators) = {len(bg)}")
    print("[fig 1] enrichment + length gradient")
    figure_enrichment_and_length(bg)
    print("[fig 2] 4-set Venns")
    figure_venn(bg)
    print(f"\nFigures written to {ROOT}")


if __name__ == "__main__":
    main()
