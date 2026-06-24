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
