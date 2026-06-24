# Impact-of-gene-regulatory-network-topology-on-heterogeneity-restoration
Contains Codes for SDE Simulations and Data analysis, and, Plot Generations.

# adjacency matrices data
Input Data: The file unique_networks_table.jld2, which provides the pre-computed adjacency matrices for the gene regulatory networks.

# code run_selected_net.jl : Purpose: Simulates stochastic 3-species gene networks and detects heterogeneous states using Gaussian Mixture Models.
1. Dependencies: Julia 1.10+, Sundials (for ODEs), DifferentialEquations (for SDEs)and a Python environment with scikit-learn
2. Outputs: multimodal_search_selected_1.h5 (trajectories) and multimodal_results_selected_1.csv (summary statistics)
3. Please ensure a valid unique_networks_table.jld2  (contains the adjacency matrix of 3 node gene topologies) file is located in the working directory.
4. Use the command line to execute the file while enabling multi-threading: julia -t auto script.jl
5. Network Size: The code defaults to DESIRED_NET_IDS = 1:100. Warning: Attempting to run all 3,410 networks at once will likely crash your system due to memory exhaustion; please run the script in smaller, manageable batches (e.g., 100 at a time, adjust accordingly) by updating this range.
6. Parameter Sampling: Uses Np = 1000 parameter sets, with Nt = 500 trials per parameter set
7. Noise Level: Controlled by the σ0 constant; update this value in the script to change the stochastic intensity.
   
Implementation & Flags:
1. Threading Flag: Please launch the script with the -t auto (or -t N) flag to enable Julia’s multi-threading for the @threads macro.
2. Thread Safety:  scikit-learn is wrapped in a py_lock because Python functions are not thread-safe.
3. The script internally enforces single-threading for NumPy and MKL (via os.environ flags). This prevents the "nested parallelism" performance bottleneck, where multiple threads compete for CPU resources.
4. The script disables the Garbage Collector during heavy simulations to maintain speed. If you run out of memory, reduce Np (parameter sets) or Nt (trials).
