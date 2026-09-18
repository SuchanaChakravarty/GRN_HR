"""
run_reproduce_pipeline.py
==============================================================================
One command that regenerates every current reproduction figure -- the hr_,
feedback_, mh_, dyn_, overshoot_ and parsimony_ series -- from the filtered
results files and the network topology, with nothing else derived on disk.

INPUTS (the only things this pipeline reads)
    network_data.json                         topology: 3,411 signed 3-node motifs
    summary_result_noise1_frac10_opicFiltered.csv     sigma 1.0   frac10 + op_ic   <- canonical
    summary_result_noise02_frac10_opicFiltered.csv    sigma 0.2   frac10 + op_ic
    summary_result_noise5_frac10_opicFiltered.csv     sigma 5.0   frac10 + op_ic
    summary_result_noise1_frac10.csv          sigma 1.0   frac10 only
    summary_result_noise02_frac10.csv         sigma 0.2   frac10 only
    summary_result_noise5_frac10.csv          sigma 5.0   frac10 only

frac_data/ is NOT an input. Until 2026-09-17 the overshoot figures read the per-set
transients from frac_data/overshoot_results_net_*.csv; they now read the results
files' own overshoot_ratio / overshoot_time / overshoot_state columns, which were
verified to carry identical information (existence agrees 100.0000% over 467,885
rows; ratio and peak time are numerically identical; the early/late label is exactly
overshoot_time > 1000). That removes a 32 MB second input and a three-key merge.

The superseded 5%-floor baseline (summary_result_noise1_0817.csv) is NOT read by any
step; it survives only as the upstream input to filter_min_fraction.py, which together
with filter_op_ic_reproducible.py produced the six files above. Those two filters are
PROVENANCE, not pipeline steps -- run them only to rebuild the inputs themselves.

Which inputs are actually consumed: the 30 figures come from the three _opic files
(sigma 1.0 for everything, plus sigma 0.2 and 5.0 for the two extra parsimony panels)
alone. The three frac10-only files are checked for presence because they are the
op_ic filter's denominator and the op_ic reproducibility analysis reads them; no
figure in these six series depends on them. They are declared so the input set is the
complete, self-consistent one.

ORDERING that matters (everything else is independent):
  * The traversal (default) dynamics run must precede the fluct and overshoot runs.
    visualize_stability_dynamics reads dyn_stability_ols<suffix>.csv WITHOUT the
    metric tag for its settling-time panel -- settling time does not depend on which
    second metric is paired with it -- so that file must already exist.
  * reproduce_noise must precede reproduce_stvar_ols, which reads
    nz_stvar_per_network<suffix>.csv.

Usage:
    python run_reproduce_pipeline.py                # everything (~10-20 min)
    python run_reproduce_pipeline.py --list         # show the plan, run nothing
    python run_reproduce_pipeline.py --dry-run      # same, with resolved env
    python run_reproduce_pipeline.py --skip-heavy   # omit the 3 parsimony fits
    python run_reproduce_pipeline.py --only dyn     # steps whose name contains "dyn"
==============================================================================
"""
import argparse
import os
import subprocess
import sys
import time

RES = "results/reproduce"
OPIC1 = "summary_result_noise1_frac10_opicFiltered.csv"
OPIC02 = "summary_result_noise02_frac10_opicFiltered.csv"
OPIC5 = "summary_result_noise5_frac10_opicFiltered.csv"
F10 = ["summary_result_noise1_frac10.csv", "summary_result_noise02_frac10.csv",
       "summary_result_noise5_frac10.csv"]
# REQUIRED is exactly what the 30 figures read. The three frac10-only files are the
# op_ic filter's denominator and feed no figure here, so they are OPTIONAL: their
# absence is reported, not fatal, and a repository packaged with package_inputs.py
# (default --minimal) legitimately ships without them.
REQUIRED = ["network_data.json", OPIC1, OPIC02, OPIC5]
OPTIONAL = F10

# Every step: (name, script, extra env, figures it must produce).
# env is layered on os.environ; HR_RESULTS/HR_POOL/DYN_METRIC2/NOISE_SET are the only
# switches repro_common reads. The defaults are already the canonical ones, so most
# steps need no env at all -- they are spelled out anyway, so the plan is explicit.
CANON = {"HR_RESULTS": OPIC1, "HR_POOL": "connected"}
STEPS = [
    ("features:dictionary", "reproduce_feature_dictionary.py", CANON, []),

    ("hr:existence", "reproduce_hr_composition.py", CANON,
     ["hr_composition_frac10_opic"]),
    ("hr:robustness", "reproduce_hr_robustness_composition.py", CANON,
     ["hr_robustness_composition_frac10_opic"]),

    ("feedback:roles", "reproduce_feedback_roles.py", CANON,
     ["feedback_roles_overall_polarity_frac10_opic",
      "feedback_roles_forest_frac10_opic",
      "feedback_roles_barchart_frac10_opic"]),

    ("mh:motif-forest", "reproduce_mh_table_standalone.py", CANON,
     ["mh_motif_forest_frac10_opic"]),
    ("mh:motif-forest-fullpool", "reproduce_mh_table_standalone.py",
     {**CANON, "HR_POOL": "full"}, ["mh_motif_forest_frac10_opic_full"]),

    # --- dynamics: traversal FIRST (writes the untagged settling-time OLS) --------
    ("dyn:fit-traversal", "reproduce_stability_dynamics.py",
     {**CANON, "DYN_METRIC2": "traversal"}, []),
    ("dyn:draw-traversal", "visualize_stability_dynamics.py",
     {**CANON, "DYN_METRIC2": "traversal"},
     ["dyn_perloop_bars_frac10_opic", "dyn_rate_forest_frac10_opic",
      "dyn_dist_violin_frac10_opic", "dyn_rate_scatter_frac10_opic",
      "dyn_within_corr_hist_frac10_opic"]),

    ("dyn:fit-fluct", "reproduce_stability_dynamics.py",
     {**CANON, "DYN_METRIC2": "fluct"}, []),
    ("dyn:draw-fluct", "visualize_stability_dynamics.py",
     {**CANON, "DYN_METRIC2": "fluct"},
     ["dyn_perloop_bars_fluct_frac10_opic", "dyn_rate_forest_fluct_frac10_opic",
      "dyn_dist_violin_fluct_frac10_opic", "dyn_rate_scatter_fluct_frac10_opic",
      "dyn_within_corr_hist_fluct_frac10_opic"]),

    ("dyn:fit-overshoot", "reproduce_stability_dynamics.py",
     {**CANON, "DYN_METRIC2": "overshoot"}, []),
    ("dyn:draw-overshoot", "visualize_stability_dynamics.py",
     {**CANON, "DYN_METRIC2": "overshoot"},
     ["dyn_perloop_bars_overshoot_frac10_opic", "dyn_rate_forest_overshoot_frac10_opic",
      "dyn_dist_violin_overshoot_frac10_opic", "dyn_rate_scatter_overshoot_frac10_opic",
      "dyn_within_corr_hist_overshoot_frac10_opic"]),
    ("dyn:overshoot-violin", "visualize_overshoot_violin.py", CANON,
     ["dyn_overshoot_violin_frac10_opic"]),

    # --- noise thread: only needed here to feed the settling-variability OLS ------
    ("noise:fit", "reproduce_noise.py", {**CANON, "NOISE_SET": "frac10_opic"}, []),
    ("dyn:stvar-ols", "reproduce_stvar_ols.py",
     {**CANON, "NOISE_SET": "frac10_opic"}, ["dyn_stvar_perloop_bars_frac10_opic"]),

    ("overshoot:by-group", "explore_overshoot.py", CANON,
     ["overshoot_by_group_frac10_opic", "overshoot_perloop_frac10_opic",
      "overshoot_timing_relative_frac10_opic"]),

    # --- parsimony: one fit + one figure per noise level (compute-heavy) ----------
    ("parsimony:fit-noise1", "reproduce_parsimony_prediction.py", CANON, []),
    ("parsimony:draw-noise1", "visualize_parsimony_prediction.py", CANON,
     ["parsimony_prediction_overview_frac10_opic"]),
    ("parsimony:fit-noise02", "reproduce_parsimony_prediction.py",
     {**CANON, "HR_RESULTS": OPIC02}, []),
    ("parsimony:draw-noise02", "visualize_parsimony_prediction.py",
     {**CANON, "HR_RESULTS": OPIC02}, ["parsimony_prediction_overview_noise02_frac10_opic"]),
    ("parsimony:fit-noise5", "reproduce_parsimony_prediction.py",
     {**CANON, "HR_RESULTS": OPIC5}, []),
    ("parsimony:draw-noise5", "visualize_parsimony_prediction.py",
     {**CANON, "HR_RESULTS": OPIC5}, ["parsimony_prediction_overview_noise5_frac10_opic"]),
]
HEAVY = ("parsimony:",)


def preflight():
    # resolve_input finds either the full-width CSV in the project root or the
    # slimmed, gzipped copy under data/ that package_inputs.py writes
    from repro_common import resolve_input
    missing = [f for f in REQUIRED if not os.path.isfile(resolve_input(f))]
    absent_opt = [f for f in OPTIONAL if not os.path.isfile(resolve_input(f))]
    if absent_opt and not missing:
        print(f"note: {len(absent_opt)} optional frac10-only file(s) not present "
              f"-- no figure in this pipeline needs them:")
        for f in absent_opt:
            print(f"    {f}")
        print()
    if missing:
        print("MISSING INPUTS -- the pipeline cannot run:")
        for m in missing:
            print(f"    {m}")
        print("\nThe six summary files come from filter_min_fraction.py followed by")
        print("filter_op_ic_reproducible.py.")
        sys.exit(1)
    os.makedirs(RES, exist_ok=True)


def run(steps, dry):
    t_all = time.time()
    rows, produced = [], 0
    for i, (name, script, env, figs) in enumerate(steps, 1):
        shown = " ".join(f"{k}={v}" for k, v in sorted(env.items()))
        print(f"\n[{i}/{len(steps)}] {name}\n    {shown} python {script}", flush=True)
        if dry:
            rows.append((name, 0.0, len(figs)))
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, script], env={**os.environ, **env})
        dt = time.time() - t0
        if r.returncode != 0:
            print(f"\nFAILED at step {i} ({name}) after {dt:.0f}s -- stopping.")
            sys.exit(r.returncode)
        for f in figs:                       # declared outputs must exist, both formats
            for ext in ("png", "svg"):
                p = f"{RES}/{f}.{ext}"
                if not os.path.isfile(p):
                    print(f"\nFAILED: {name} did not write {p}")
                    sys.exit(2)
        produced += len(figs)
        rows.append((name, dt, len(figs)))
        print(f"    ok  {dt:6.1f}s  {len(figs)} figure(s)")

    print("\n" + "=" * 62)
    print(f"{'step':28s} {'seconds':>9s} {'figures':>9s}")
    for n, dt, k in rows:
        print(f"{n:28s} {dt:9.1f} {k:9d}")
    print("=" * 62)
    print(f"{'TOTAL':28s} {time.time() - t_all:9.1f} {produced:9d}"
          f"   (x2 files: png + svg)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--dry-run", action="store_true", help="print resolved env, run nothing")
    ap.add_argument("--skip-heavy", action="store_true", help="omit the parsimony fits")
    ap.add_argument("--only", metavar="SUBSTR", help="only steps whose name contains SUBSTR")
    a = ap.parse_args()

    steps = STEPS
    if a.skip_heavy:
        steps = [s for s in steps if not s[0].startswith(HEAVY)]
    if a.only:
        steps = [s for s in steps if a.only in s[0]]
    if not steps:
        print("no steps matched")
        sys.exit(1)

    if a.list:
        total = sum(len(s[3]) for s in steps)
        for n, script, _, figs in steps:
            print(f"{n:28s} {script:38s} {len(figs)} fig")
        print(f"\n{len(steps)} steps, {total} figures")
        sys.exit(0)

    preflight()
    run(steps, a.dry_run)
