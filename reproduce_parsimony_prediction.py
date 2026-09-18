"""
reproduce_parsimony_prediction.py
==============================================================================
Standalone reproduction of the DATA behind the two parsimony presentation figures
    parsimony_03_vs_null      (robustness regressor: R^2 & Spearman)
    parsimony_clf_03_vs_null  (existence classifier: ROC-AUC & minority PR-AUC)

Claim reproduced: a small, biologically interpretable feature set (density gate +
feedback taxonomy) already predicts multimodality well above chance, while the full
topology set is near-perfect. Five bars per panel:
    density  ->  feedback  ->  density+feedback  ->  full  ->  null (shuffled)

Method (identical to Phases 21/22 so the numbers match):
  * RandomForest (300 trees) with repeated (15x) 5-fold CV; a fresh shuffled
    partition per repeat (random_state = repeat index) -> pooled-OOF metric per
    repeat -> report mean +/- SD across repeats.
  * Robustness target = multimodal_rate: R^2 and Spearman.
  * Existence target = label (>=1 multimodal run): ROC-AUC and MINORITY
    (non-multimodal, prevalence ~0.10) PR-AUC -- the honest metric at 90/10 imbalance.
  * Null = labels/rate shuffled, same repeated-CV on the density+feedback features.

Depends ONLY on the two raw files (network_data.json, the results CSV) via
repro_common.build_feature_table -- NO dependence on results/features.csv.

Outputs (results/reproduce/, NO figures -- see visualize_parsimony_prediction.py):
    parsimony_presentation.csv       set, n_features, cv_r2(+sd), spearman(+sd)
    parsimony_clf_presentation.csv   set, n_features, roc_auc(+sd), pr_auc_minority(+sd)
    parsimony_reference.csv          per-metric chance/baseline levels for the plot

NOTE: intentionally compute-heavy (repeated CV over the 63-feature full model on
3,411 networks, both tasks + nulls) -> a few minutes. Run from the project root:
    python reproduce_parsimony_prediction.py
==============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score

from repro_common import build_feature_table, CONNECTED, OUT_SUFFIX
from motif_lib import DEPRECATED_FEATURE_COLS

warnings.filterwarnings("ignore")
os.makedirs("results/reproduce", exist_ok=True)
RNG = 42          # RandomForest seed (matches Phases 21/22)
N_REP = 15        # repeated 5-fold CV -> mean +/- SD

# =============================================================================
# 0. Build the full feature table from the two raw files (no derived inputs)
# =============================================================================
df = build_feature_table(connected=CONNECTED)

# --- feature sets (identical to Phases 21/22) --------------------------------
GATE_DENS = ["n_edges"]                                   # density gate only
FB = ["fb_pos_1node", "fb_neg_1node", "fb_pos_2node",
      "fb_neg_2node", "fb_pos_3node", "fb_neg_3node"]     # feedback taxonomy (polarity x length)

META = ["net_id", "label", "multimodal_count", "multimodal_rate",
        "label_t05", "label_t10", "label_t25"]
DROP = set(DEPRECATED_FEATURE_COLS) | {"edge_density", "n_zero"} | \
       {c for c in df.columns if c.startswith("has_")} | \
       {"fb_pos_feedback", "fb_neg_feedback", "fb_total_feedback"}
FULL = [c for c in df.columns if c not in META and c not in DROP]   # 63 features (FFL-free)

SETS = {                       # the five presentation bars (null handled separately)
    "density":          GATE_DENS,
    "feedback":         FB,
    "density+feedback": GATE_DENS + FB,
    "full":             FULL,
}
y_rate = df["multimodal_rate"].values
y_lab  = df["label"].values
prevalence = float(1 - y_lab.mean())                      # minority (non-multimodal) base rate
print(f"{len(df)} networks | full = {len(FULL)} features | "
      f"multimodal prevalence = {y_lab.mean():.3f} (minority base rate {prevalence:.3f})")

# =============================================================================
# 1. Repeated 5-fold CV helpers (pooled-OOF metric per repeat)
# =============================================================================
def repeated_regression(cols, target):
    r2s, sps = [], []
    for r in range(N_REP):
        p = cross_val_predict(RandomForestRegressor(300, n_jobs=-1, random_state=RNG),
                              df[cols].values, target, n_jobs=1,
                              cv=KFold(5, shuffle=True, random_state=r))
        r2s.append(r2_score(target, p))
        sps.append(stats.spearmanr(target, p).correlation)
    return np.mean(r2s), np.std(r2s), np.mean(sps), np.std(sps)

def repeated_classification(cols, target):
    aucs, prms = [], []
    for r in range(N_REP):
        p = cross_val_predict(RandomForestClassifier(300, n_jobs=-1, random_state=RNG),
                              df[cols].values, target, method="predict_proba", n_jobs=1,
                              cv=StratifiedKFold(5, shuffle=True, random_state=r))[:, 1]
        aucs.append(roc_auc_score(target, p))
        prms.append(average_precision_score(1 - target, 1 - p))   # MINORITY PR-AUC
    return np.mean(aucs), np.std(aucs), np.mean(prms), np.std(prms)

# =============================================================================
# 2. Robustness regressor (target = multimodal_rate)
# =============================================================================
print("\n=== Robustness regressor: repeated 5-fold CV R^2 / Spearman ===")
reg_rows = []
for name, cols in SETS.items():
    r2m, r2sd, spm, spsd = repeated_regression(cols, y_rate)
    reg_rows.append({"set": name, "n_features": len(cols), "cv_r2": r2m, "cv_r2_sd": r2sd,
                     "spearman": spm, "spearman_sd": spsd})
    print(f"  {name:16s} ({len(cols):2d} feat): R2={r2m:.3f}+/-{r2sd:.3f}  "
          f"Spearman={spm:.3f}+/-{spsd:.3f}")

# null: shuffle the rate, evaluate on density+feedback features
rng = np.random.RandomState(RNG)
Xnull_cols = GATE_DENS + FB
nr2, nsp = [], []
for r in range(N_REP):
    ys = rng.permutation(y_rate)
    p = cross_val_predict(RandomForestRegressor(200, n_jobs=-1, random_state=RNG),
                          df[Xnull_cols].values, ys, n_jobs=1,
                          cv=KFold(5, shuffle=True, random_state=r))
    nr2.append(r2_score(ys, p)); nsp.append(stats.spearmanr(ys, p).correlation)
reg_rows.append({"set": "shuffled", "n_features": len(Xnull_cols),
                 "cv_r2": np.mean(nr2), "cv_r2_sd": np.std(nr2),
                 "spearman": np.mean(nsp), "spearman_sd": np.std(nsp)})
print(f"  {'shuffled':16s} ({len(Xnull_cols):2d} feat): R2={np.mean(nr2):.3f}+/-{np.std(nr2):.3f}  "
      f"Spearman={np.mean(nsp):.3f}+/-{np.std(nsp):.3f}")
pd.DataFrame(reg_rows).to_csv(f"results/reproduce/parsimony_presentation{OUT_SUFFIX}.csv", index=False)

# =============================================================================
# 3. Existence classifier (target = label)
# =============================================================================
print("\n=== Existence classifier: repeated 5-fold CV ROC-AUC / minority PR-AUC ===")
clf_rows = []
for name, cols in SETS.items():
    aucm, aucsd, prmm, prmsd = repeated_classification(cols, y_lab)
    clf_rows.append({"set": name, "n_features": len(cols), "roc_auc": aucm, "roc_auc_sd": aucsd,
                     "pr_auc_minority": prmm, "pr_auc_minority_sd": prmsd})
    print(f"  {name:16s} ({len(cols):2d} feat): ROC-AUC={aucm:.3f}+/-{aucsd:.3f}  "
          f"PR-AUC(min)={prmm:.3f}+/-{prmsd:.3f}")

rng = np.random.RandomState(RNG)
nauc, nprm = [], []
for r in range(N_REP):
    ys = rng.permutation(y_lab)
    p = cross_val_predict(RandomForestClassifier(200, n_jobs=-1, random_state=RNG),
                          df[Xnull_cols].values, ys, method="predict_proba", n_jobs=1,
                          cv=StratifiedKFold(5, shuffle=True, random_state=r))[:, 1]
    nauc.append(roc_auc_score(ys, p)); nprm.append(average_precision_score(1 - ys, 1 - p))
clf_rows.append({"set": "shuffled", "n_features": len(Xnull_cols),
                 "roc_auc": np.mean(nauc), "roc_auc_sd": np.std(nauc),
                 "pr_auc_minority": np.mean(nprm), "pr_auc_minority_sd": np.std(nprm)})
print(f"  {'shuffled':16s} ({len(Xnull_cols):2d} feat): ROC-AUC={np.mean(nauc):.3f}+/-{np.std(nauc):.3f}  "
      f"PR-AUC(min)={np.mean(nprm):.3f}+/-{np.std(nprm):.3f}")
pd.DataFrame(clf_rows).to_csv(f"results/reproduce/parsimony_clf_presentation{OUT_SUFFIX}.csv", index=False)

# =============================================================================
# 4. Reference (chance / baseline) levels for the visualization
# =============================================================================
pd.DataFrame([
    {"metric": "cv_r2",           "chance": 0.0},
    {"metric": "spearman",        "chance": 0.0},
    {"metric": "roc_auc",         "chance": 0.5},
    {"metric": "pr_auc_minority", "chance": prevalence},
]).to_csv(f"results/reproduce/parsimony_reference{OUT_SUFFIX}.csv", index=False)

print(f"\n[pool: {'connected-only' if CONNECTED else 'full'}]  Saved -> "
      f"results/reproduce/parsimony_presentation{OUT_SUFFIX}.csv, "
      f"parsimony_clf_presentation{OUT_SUFFIX}.csv, parsimony_reference{OUT_SUFFIX}.csv")
