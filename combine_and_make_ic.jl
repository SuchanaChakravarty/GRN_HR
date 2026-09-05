#=
================================================================================
 combine_and_make_ic.jl

 PURPOSE
 -------
 Two-step post-processing on the output of `run_selected_net.jl`:

   1. combine_and_sort_csv() — gather all per-net `multimodal_results_net*.csv`
      files in a directory into one summary CSV, sorted by net_id.
   2. make_ic_from_far()     — from that summary, pull the far-from-IC value
      per (net_id, param_id, species) and write `ic_from_far.dat`, the file
      `run_selected_net.jl precomputed` reads back in.

 USAGE
 -----
 Edit the constants below, then:
     julia combine_and_make_ic.jl
================================================================================
=#

using CSV, DataFrames

# ==============================================================
# Config — edit these for your run
# ==============================================================
const DATA_DIR    = raw"F:\HR_Project\noiseo2" #change it to your own file path
const PATTERN     = "multimodal_results_net"
const SORT_COLUMN = :net_id
const SUMMARY_CSV = joinpath(DATA_DIR, "summary_result_noiseo2.csv")
const OUT_DAT     = joinpath(DATA_DIR, "ic_from_far_noiseo2.dat")

# ==============================================================
# Step 1: combine + sort all per-net result CSVs into one summary CSV
# ==============================================================
function combine_and_sort_csv(dir::AbstractString, pattern::String, sort_column::Symbol)
    all_files = readdir(dir)
    target_files = filter(f -> startswith(f, pattern) && endswith(f, ".csv"), all_files)

    if isempty(target_files)
        println("Warning: No files matching '$pattern' found in $dir!")
        return DataFrame()
    end
    println("Found files: ", target_files)

    dfs = [DataFrame(CSV.File(joinpath(dir, f))) for f in target_files]
    combined_df = vcat(dfs...)

    if string(sort_column) in names(combined_df)
        sort!(combined_df, sort_column)
        println("Successfully sorted by $sort_column.")
    else
        println("Warning: Column $sort_column not found. Returning unsorted.")
    end

    return combined_df
end

# ==============================================================
# Step 2: extract far-from-IC values -> ic_from_far.dat
# ==============================================================
function make_ic_from_far(df::DataFrame, out_dat::AbstractString)
    sub = select(df, [:net_id, :param_id, :far_from_ic_x1, :far_from_ic_x2, :far_from_ic_x3])
    sort!(sub, [:net_id, :param_id])

    open(out_dat, "w") do io
        println(io, "# net_id param_id far_x1 far_x2 far_x3")
        for r in eachrow(sub)
            x1 = ismissing(r.far_from_ic_x1) ? NaN : r.far_from_ic_x1
            x2 = ismissing(r.far_from_ic_x2) ? NaN : r.far_from_ic_x2
            x3 = ismissing(r.far_from_ic_x3) ? NaN : r.far_from_ic_x3
            println(io, "$(r.net_id) $(r.param_id) $x1 $x2 $x3")
        end
    end
    println("wrote $(nrow(sub)) rows, $(length(unique(sub.net_id))) nets -> $out_dat")
end

# ==============================================================
# Execution
# ==============================================================
multimodal_full = combine_and_sort_csv(DATA_DIR, PATTERN, SORT_COLUMN)

if nrow(multimodal_full) > 0
    CSV.write(SUMMARY_CSV, multimodal_full)
    println("Combined and sorted file saved to: $SUMMARY_CSV")
    make_ic_from_far(multimodal_full, OUT_DAT)
else
    println("No data combined — skipping IC extraction.")
end
