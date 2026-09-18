"""
Standalone reproduction of the density-controlled (Mantel-Haenszel) motif table
for PFL / NFL / coherent-FFL / incoherent-FFL, from ONLY two raw inputs:
    1. network_data.json
    2. multimodal_results_combined_sorted.csv  (or .xlsx)

All shared logic lives in repro_common.py (table assembly + crude/MH/CMH stats);
the only other code dependency is motif_lib.py. See REPRODUCE_MH_MOTIF_TABLE.md
for the step-by-step walkthrough.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from repro_common import build_network_table, crude_and_mh, savefig, CONNECTED, OUT_SUFFIX

os.makedirs("results/reproduce", exist_ok=True)

# (display name, presence flag, colour)
MOTIFS = [
    ("PFL (any positive feedback)", "has_fb_pos_feedback", "#2196F3"),
    ("NFL (any negative feedback)", "has_fb_neg_feedback", "#1565C0"),
    ("Coherent FFL",                "has_ffl_coherent",    "#FB8C00"),
    ("Incoherent FFL",              "has_ffl_incoherent",  "#E64A19"),
]

# --- build table from the two raw files, compute stats ----------------------
df = build_network_table(connected=CONNECTED)
print(f"[pool: {'connected-only' if CONNECTED else 'full'}]  {len(df)} networks "
      f"({df['label'].sum()} multimodal, {(df['label'] == 0).sum()} non-multimodal)\n")

rows = [{"motif": name, **crude_and_mh(df, flag)} for name, flag, _ in MOTIFS]
tab = pd.DataFrame(rows)

show = tab.copy()
show["crude_logOR_CI"] = show.apply(
    lambda r: f"[{r['crude_ci_low']:+.3f}, {r['crude_ci_high']:+.3f}]", axis=1)
show["MH_logOR_CI"] = show.apply(
    lambda r: f"[{r['mh_ci_low']:+.3f}, {r['mh_ci_high']:+.3f}]", axis=1)
show["CMH_p"] = show["cmh_p"].map(lambda p: f"{p:.2e}")
cols = ["motif", "n_present", "crude_log_or", "crude_logOR_CI", "mh_log_or", "MH_logOR_CI",
        "cmh_chi2", "CMH_p", "breslow_day_p", "n_strata", "strata"]
pd.set_option("display.width", 200, "display.max_columns", 20)
print(show[cols].to_string(index=False))

tab.to_csv(f"results/reproduce/mh_motif_table{OUT_SUFFIX}.csv", index=False)

# --- forest plot ------------------------------------------------------------
colors = {name: c for name, _, c in MOTIFS}
fig, ax = plt.subplots(figsize=(9, 4.6))
ypos = np.arange(len(tab))[::-1]
for yp, (_, r) in zip(ypos, tab.iterrows()):
    c = colors[r["motif"]]
    ax.plot([r["mh_ci_low"], r["mh_ci_high"]], [yp, yp], color=c, lw=2.5, zorder=1)
    ax.scatter(r["mh_log_or"], yp, color=c, s=110, zorder=2, edgecolor="white", linewidth=1.2)
    ax.text(r["mh_ci_high"] + 0.06, yp, f"p={r['cmh_p']:.1e}".replace("-", "−"), va="center",
            fontsize=8, color="gray")
ax.axvline(0, color="black", lw=0.9, ls="--")
ax.set_yticks(ypos); ax.set_yticklabels(tab["motif"], fontsize=10)
ax.set_xlabel("Density-controlled (Mantel-Haenszel) log odds ratio for multimodality\n"
              "stratified on n_edges (95% CI; annotation = CMH p)", fontsize=10)
ax.set_title("Feedback vs Feedforward Motif Contributions to Multimodality\n"
             "(density-controlled; right = promotes, left = suppresses)",
             fontsize=12, fontweight="bold")
ax.legend(handles=[mpatches.Patch(color="#1976D2", label="Feedback (PFL/NFL)"),
                   mpatches.Patch(color="#F4511E", label="Feedforward (FFL)")],
          fontsize=9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig(f"results/reproduce/mh_motif_forest{OUT_SUFFIX}", bbox_inches="tight")
print(f"\nSaved: results/reproduce/mh_motif_table{OUT_SUFFIX}.csv + "
      f"mh_motif_forest{OUT_SUFFIX}.{{png,svg}}")
