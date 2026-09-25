# Impact-of-gene-regulatory-network-topology-on-heterogeneity-restoration
Contains Codes for SDE Simulations and Data analysis, and, Plot Generations.

# adjacency matrices generating code 
1. Purpose: Generates all unique 3 X 3 gene regulatory network structures (adjacency matrices) by filtering out isomorphic duplicates.
2. Requirements: Jupyter Notebook, Julia, Combinatorics, and JLD2 packages.
3. Output: unique_networks_table.jld2 (a file containing the structural library for your simulations).
4. How to run: Open the .ipynb file in Jupyter, select the Julia kernel, and run all cells.

Implementation & Flags:
1. Canonicalization: The script ensures that isomorphic networks (networks that are identical if nodes are re-labeled) are treated as the same, reducing the dataset to only unique functional structures.
2. Data Format: The output is saved as an array of NamedTuples within the JLD2 file, making it easy to load as a structured table in your main simulation script.
3. Performance: The script generates all 3^9 (19,683) possible 3x3 matrices with values in {-1, 0, 1}.The canonicalization process is computationally intensive as it checks all 3! = 6 permutations for every generated network.

# adjacency matrices data
Input Data: The file unique_networks_table.jld2, which provides the pre-computed adjacency matrices for the gene regulatory networks.

# code run_selected_net.jl : Purpose: Simulates stochastic 3-species gene networks and detects heterogeneous (multimodal) states using Gaussian Mixture Models. Runs in one of two modes from a single file :default Initial Condition (IC) (computed from the ODE steady state) or precomputed IC (read from a .dat file).
1. Dependencies: Julia 1.10+, Sundials (ODEs), DifferentialEquations (SDEs), a Python environment with scikit-learn.
2. multimodal_search_net_1[_op_ic].h5 (trajectories), fraction_data_net_1[_op_ic].h5 (2 different state fractions), multimodal_results_net_1[_op_ic].csv and overshoot_results_net_1[_op_ic].csv (summary stats), progress_new_1[_op_ic].log (live progress — check this during long runs). The _op_ic suffix is added automatically in precomputed mode.
3. Please ensure a valid unique_networks_table.jld2 (adjacency matrices of 3-node gene topologies) is in the working directory. If it's missing, the script does not error — it falls back to a single placeholder network (id=8), which is fine for a smoke test but easy to miss if you forgot to add the real file. In precomputed mode, also ensure ic_from_far.dat is present — this file is produced by combine_and_make_ic.jl from a prior default-mode run's output.
4. Run modes (command line argument selects the mode):
   i.  julia -t auto run_selected_net.jl → default IC, all Np parameter sets per network.
   ii. julia -t auto run_selected_net.jl precomputed → IC read from ic_from_far.dat, only the (net, param) pairs listed there are run.
5. Network Size: defaults to DESIRED_NET_IDS = 1:200 (line 457). Warning: running all ~3,410 networks at once will likely crash your system from memory exhaustion — run in smaller batches (e.g. 100–200 at a time) by editing this range.
6. Parameter Sampling: Np = 1000 parameter sets, Nt = 500 trials per parameter set (line 91, const Np, Nic, Nt = 1000, 1, 500).
7. Noise Level: controlled by the σ0 constant (line 497, const σ0 = 1.0); update it to change stochastic intensity.
8. Output meaning: only (net, param) pairs that pass the bimodality test appear in the summary CSV — a smaller-than-expected row count doesn't necessarily mean something went wrong; it may mean fewer pairs turned out to be multimodal.
9. Runtime/disk: budget accordingly — runtime scales with Np × Nt × (number of networks), and each .h5 file holds full per-trial trajectories, so batch size should be chosen based on your available memory and disk space.
   
Implementation & Flags:
1. Threading Flag: launch with -t auto (or -t N) to enable Julia's multithreading for the @threads macro.
2. Thread Safety: scikit-learn calls are wrapped in a py_lock since Python isn't thread-safe.
3. The script forces single-threading for NumPy/MKL (via os.environ) to avoid nested-parallelism contention with Julia's threads.
4. The garbage collector is disabled during the heavy simulation loop to maintain speed. If you run out of memory, reduce Np or Nt.

# combine_and_make_ic.jl : Post-processing step for run_selected_net.jl output. Combines per-network result CSVs into one summary, then extracts the far-from-IC value per (net, param, species) to build the ic_from_far.dat file needed for a precomputed-IC re-run.
1. Dependencies: Julia 1.10+, CSV.jl, DataFrames.jl. No Python/threading needed.
2. Inputs: a directory containing multimodal_results_net*.csv files (the summary CSV output of run_selected_net.jl, one or more batches). Each file must contain at least the columns net_id, param_id, far_from_ic_x1, far_from_ic_x2, far_from_ic_x3 — if this schema changes upstream, this script breaks.
3. Outputs:
   i.  a combined, net_id-sorted summary CSV (all batches merged into one file)
   ii. ic_from_far.dat — whitespace-delimited, columns net_id param_id far_x1 far_x2 far_x3, missing values written as NaN. This is the file  run_selected_net.jl precomputed reads.
4. Run with: julia combine_and_make_ic.jl (single-threaded, no flags needed).
5. Config: edit the four constants at the top of the script — DATA_DIR, PATTERN, SUMMARY_CSV, OUT_DAT — to point at your batch output folder and desired file names. DATA_DIR uses Julia's raw"..." string, written for a Windows-style path (backslashes); on Linux/Mac, use a normal string with forward slashes instead, e.g. "/home/user/HR_Project/noiseo2".
6. If no matching CSVs are found in DATA_DIR, the script prints a warning and skips IC extraction rather than erroring out.
7. Overwrite behavior: both the summary CSV and the .dat file are overwritten without confirmation if they already exist at SUMMARY_CSV/OUT_DAT — back up first if you need to keep a prior version.

# filter_by_min_fraction.jl : Step 3 of the pipeline. Removes (net_id, param_id) rows where the split between the two states is too imbalanced to trust as real bimodality (default: one state < 10%), and reports the sensitivity/specificity evidence behind that cutoff.
1. Dependencies: Julia, CSV, DataFrames, Printf, Plots.
2. Run after: run_selected_net.jl → combine_and_make_ic.jl. Needs their outputs (summary CSV + overshoot_results_net_*.csv).
3. Run with: julia filter_by_min_fraction.jl.
4. Config (edit at top of script): SUMMARY_CSV, OVERSHOOT_DIR, N_NETWORKS, OUT_DIR, THRESHOLD.
5. Main output to use downstream: summary_clean.csv. Also writes why_this_threshold.txt, combined_table.csv, rows_removed.csv, threshold_sweep_plot.png.
6. For precomputed-mode results, swap which overshoot-filename line is commented (near line 65, _op_ic suffix).
   
# Fig 2 Plotting code : Fig2_Plot.ipynb
A Julia-based pipeline to process network simulation data, detect structural motifs, and visualize the impact of feedback and Feedforward loops on HR.
## Features
* **Data Load**: Use the summary CSV files.
* **Visualization**: Generates trajectory plots and GMM-based density distributions (Fig 2B).
* **Motif Analysis**: Detects feedback loops (positive, negative: Self-loops, pairwise loops, 3 node cyclic loop) and incoherent feed-forward loops (iFFL) (Fig 2C-2H).
* **Statistical Inference**: Performs Fisher's exact tests and computes Odds Ratios to analyze motif-HR associations, visualized via heatmaps and stacked bar plots and edgewise network analysis.
  ## Requirements
* **Julia** with `CSV`, `DataFrames`, `JLD2`, `HDF5`, `Sundials`, `DifferentialEquations`, `LatinHypercubeSampling`, `SpecialFunctions`, `PyCall` (for scikit-learn GMM fitting).
