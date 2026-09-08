#=
================================================================================
 filter_by_min_fraction.jl

 PURPOSE
 -------
 Third step in the pipeline (after run_selected_net.jl -> combine_and_make_ic.jl).
 Flags and removes (net_id, param_id) rows whose on/off population is too
 imbalanced to trust as genuinely bimodal — i.e. one branch holds less than
 THRESHOLD (default 10%) of the final population, which is more likely to be
 a near-degenerate/borderline GMM call than a real switch. Also produces the
 sensitivity/specificity evidence used to justify that threshold choice.

 PIPELINE POSITION
 ------------------
   1) run_selected_net.jl        -> multimodal_results_net_*.csv,
                                     overshoot_results_net_*.csv (per network)
   2) combine_and_make_ic.jl     -> summary_result_*.csv (combined + sorted),
                                     ic_from_far_*.dat
   3) filter_by_min_fraction.jl  -> summary_clean.csv (THIS SCRIPT)

 REQUIRED INPUT FILES
 ---------------------
   1) summary CSV (from step 2)
        columns: net_id, param_id, stability_time, accepted_nodes
   2) overshoot_results_net_1.csv ... net_N.csv (from step 1)
        columns: net_id, param_id, on_final, off_final

 OUTPUTS (written to OUT_DIR)
 -----------------------------
   - combined_table.csv       : stability_time + min_frac + is_zero per row
   - why_this_threshold.txt   : sensitivity/specificity/Youden's J across a
                                 range of candidate cutoffs, plus the chosen
                                 THRESHOLD's stats
   - summary_flagged.csv      : full summary with a `flagged` column added
   - summary_clean.csv        : summary with flagged rows removed (use this
                                 downstream)
   - rows_removed.csv         : the rows that got flagged/removed
   - threshold_sweep_plot.png : flagged param sets vs. networks lost, swept
                                 across cutoffs, to sanity-check THRESHOLD

 USAGE
 -----
 Edit the SETTINGS block below, then: julia filter_by_min_fraction.jl
================================================================================
=#

using CSV, DataFrames, Printf, Plots

# ---- SETTINGS ----
SUMMARY_CSV   = raw"F:\HR_Project\noiseo2\summary_result_noiseo2.csv"
OVERSHOOT_DIR = raw"F:\HR_Project\noiseo2"
N_NETWORKS    = 17 # how many overshoot.csv files are there
OUT_DIR       = raw"C:\Users\Suchana\OneDrive - The University of Texas at Dallas\UTD_WORK\Suchana_UTD_Project_1\HR_Project\noise1_new\summary_result_noiseo2_clean_summary"
THRESHOLD     = 0.10
mkpath(OUT_DIR)

# ---- load summary ----
summary = CSV.read(SUMMARY_CSV, DataFrame)
rename!(summary, names(summary)[1] => :net_id)

# ---- build on/off lookup from overshoot files ----
onoff = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
for i in 1:N_NETWORKS
    f = joinpath(OVERSHOOT_DIR, "overshoot_results_net_$(i).csv")   # for normal run
    #f = joinpath(OVERSHOOT_DIR, "overshoot_results_net_$(i)_op_ic.csv")    # for op ic run
    isfile(f) || continue
    d = CSV.read(f, DataFrame)
    for r in eachrow(d)
        onoff[(r.net_id, r.param_id)] = (Float64(r.on_final), Float64(r.off_final))
    end
end

# ---- combine: min_frac + is_zero ----
combined = DataFrame(net_id=Int[], param_id=Int[], stability_time=Float64[],
                      min_frac=Float64[], is_zero=Bool[])
for row in eachrow(summary)
    ismissing(row.stability_time) && continue
    st = Float64(row.stability_time)
    isnan(st) && continue
    key = (row.net_id, row.param_id)
    haskey(onoff, key) || continue
    on_f, off_f = onoff[key]
    push!(combined, (row.net_id, row.param_id, st, min(on_f, off_f), st == 0.0))
end
CSV.write(joinpath(OUT_DIR, "combined_table.csv"), combined)

# ---- threshold evidence: sensitivity / specificity / Youden J ----
n_zero, n_nonzero = count(combined.is_zero), nrow(combined) - count(combined.is_zero)
open(joinpath(OUT_DIR, "why_this_threshold.txt"), "w") do io
    println(io, "cutoff  sensitivity  specificity  YoudenJ  n_flagged")
    for c in 0.0:0.01:0.3
        tp = count(r -> r.is_zero  && r.min_frac <= c, eachrow(combined))
        fp = count(r -> !r.is_zero && r.min_frac <= c, eachrow(combined))
        sens = tp / n_zero
        spec = (n_nonzero - fp) / n_nonzero
        @printf(io, "%.2f    %.4f       %.4f       %.4f   %d\n", c, sens, spec, sens+spec-1, tp+fp)
    end
    println(io, "\nCHOSEN THRESHOLD = ", THRESHOLD)
    tp = count(r -> r.is_zero  && r.min_frac <= THRESHOLD, eachrow(combined))
    fp = count(r -> !r.is_zero && r.min_frac <= THRESHOLD, eachrow(combined))
    all_nets  = Set(combined.net_id)
    kept_nets = Set(filter(r -> r.min_frac > THRESHOLD, combined).net_id)
    @printf(io, "sensitivity=%.4f  specificity=%.4f  rows_flagged=%d (%.2f%%)  networks_lost=%d\n",
            tp/n_zero, (n_nonzero-fp)/n_nonzero, tp+fp, 100*(tp+fp)/nrow(combined),
            length(setdiff(all_nets, kept_nets)))
end

# ---- apply threshold to full summary ----
flag = Dict((r.net_id, r.param_id) => r.min_frac <= THRESHOLD for r in eachrow(combined))
summary.flagged = [get(flag, (r.net_id, r.param_id), false) for r in eachrow(summary)]
CSV.write(joinpath(OUT_DIR, "summary_flagged.csv"), summary)
CSV.write(joinpath(OUT_DIR, "summary_clean.csv"), filter(r -> !r.flagged, summary))
CSV.write(joinpath(OUT_DIR, "rows_removed.csv"), filter(r -> r.flagged, summary))

# ---- plot: param sets flagged vs networks lost, across cutoffs ----
cutoffs = collect(0.05:0.005:0.15)
all_nets = Set(combined.net_id)
n_flagged_param_sets = Int[]
n_networks_lost = Int[]
for c in cutoffs
    push!(n_flagged_param_sets, count(v -> v <= c, combined.min_frac))
    kept_nets = Set(filter(r -> r.min_frac > c, combined).net_id)
    push!(n_networks_lost, length(setdiff(all_nets, kept_nets)))
end

p = plot(cutoffs, n_flagged_param_sets,
    label = "# parameter sets flagged (left axis)",
    linewidth = 3, color = :steelblue, marker = :circle, markersize = 4,
    xlabel = "min_frac threshold", ylabel = "number of parameter sets",
    legend = :topleft, size = (850, 550), dpi = 300,
    left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, right_margin = 12Plots.mm)

p2 = twinx(p)
plot!(p2, cutoffs, n_networks_lost,
    label = "# networks lost entirely (right axis)",
    linewidth = 3, color = :indianred, marker = :diamond, markersize = 4,
    ylabel = "number of networks lost", legend = :bottomright)

savefig(p, joinpath(OUT_DIR, "threshold_sweep_plot.png"))
display(p)

println("Done. Check: combined_table.csv, why_this_threshold.txt, summary_clean.csv, rows_removed.csv, threshold_sweep_plot.png")
