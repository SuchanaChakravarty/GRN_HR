# Restrict Python/NumPy/MKL to single thread
using PyCall
const py_lock = ReentrantLock() # python lock
py"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
"""
using JLD2, DifferentialEquations, Random, HDF5, Dates, Sundials
using LinearAlgebra, Statistics, DataFrames, CSV
using LatinHypercubeSampling
using Base.Threads
sklearn = pyimport("sklearn.mixture")
# ==============================================================
# Progress logging
# ==============================================================
const LOG_FILE = "progress_new_1.log"
function logmsg(msg)
    ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    open(LOG_FILE, "a") do f
        println(f, "[$ts] [T$(threadid())] $msg")
    end
    # The important additions:
    println("[$ts] [T$(threadid())] $msg")
    flush(stdout)
end
# ==============================================================
# Seeding constants
# ==============================================================
const BASE_SEED_CONSTANT = 20251208
const NETWORK_SEED_MULTIPLIER = 1000000
const PARAM_SEED_MULTIPLIER = 1000
# ==============================================================
# Summary table
# ==============================================================
summary_df = DataFrame(
    net_id = Int[],
    param_id = Int[],
    threshold_x1 = Float64[], threshold_x2 = Float64[], threshold_x3 = Float64[],
    stability_time = Float64[],
    cv_before_x1 = Float64[], cv_before_x2 = Float64[], cv_before_x3 = Float64[],
    cv_after_x1 = Float64[], cv_after_x2 = Float64[], cv_after_x3 = Float64[],
    mean_path = Float64[], std_path = Float64[], median_path = Float64[]
)
# ==============================================================
# Parameter ranges
# ==============================================================
const PARAM_RANGES_GENERAL = vcat(
    fill((0.001, 0.1),  3),   # b — basal production
    fill((0.01,  0.5), 3),   # d — degradation
    fill((0.1,   5.0),  9),   # K — Hill threshold
    fill((1.0,   10.0), 9)    # n — cooperativity
)

const V_CONST = 1.0            # V fixed as constant

const N_SAMPLED_PARAMS = length(PARAM_RANGES_GENERAL)
const Np, Nic, Nt = 1000, 1, 500

#Truncated lognormal sampling via inverse CDF method
using SpecialFunctions   # for erfinv
 
# Standard normal CDF: Phi(x)
function normal_cdf(x::Float64)
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))
end
 
# Standard normal inverse CDF (quantile function): Phi^{-1}(p)
function normal_invcdf(p::Float64)
    return sqrt(2.0) * erfinv(2.0 * p - 1.0)
end
 
# Sample one value from lognormal truncated to [a, b]
# via inverse CDF method 
function sample_truncated_lognormal(a::Float64, b::Float64, mu::Float64, sigma::Float64)
    # CDF of underlying normal evaluated at log(a) and log(b)
    p_low  = normal_cdf((log(a) - mu) / sigma)
    p_high = normal_cdf((log(b) - mu) / sigma)
    # Draw uniform on [p_low, p_high] then invert
    u = p_low + rand() * (p_high - p_low)
    # Clamp u to valid range to avoid numerical edge issues
    u = clamp(u, 1e-14, 1.0 - 1e-14)
    return exp(mu + sigma * normal_invcdf(u))
end


function generate_lognormal_params(num_samples, ranges; max_attempts=1000)
    num_dims = length(ranges)
    for _ in 1:max_attempts
        samples = zeros(num_samples, num_dims)
        for (j, (a, b)) in enumerate(ranges)
            # Log-space mean = geometric midpoint of bounds
            mu    = (log(a) + log(b)) / 2.0
            # Log-space sigma = range spans ~4 sigma (±2 sigma rule)
            sigma = (log(b) - log(a)) / 4.0
            for i in 1:num_samples
                samples[i, j] = sample_truncated_lognormal(
                    Float64(a), Float64(b), mu, sigma
                )
            end
        end
        samples[:, 16:24] .= ceil.(samples[:, 16:24])
        if length(unique(eachrow(samples))) == num_samples
            return samples
        end
    end
    error("Could not generate unique lognormal samples after $max_attempts attempts")
end

# ==============================================================
# Model constants
# ==============================================================
const N_SPECIES = 3
const N_OU = 3
const N_TOTAL = N_SPECIES + N_OU
const HILL_EPS = 1e-8
const MINVAL = 1e-5
@inline hill_act(x, K, n, V) = let m = max(x / V, HILL_EPS)
    (K^n * m^n) / (K^n + m^n)
end
@inline hill_inh(x, K, n, V) = let m = max(x / V, HILL_EPS)
    K^n / (K^n + m^n)
end

# Layout: p[1:3]=b, p[4:6]=d, p[7:15]=K, p[16:24]=n,
#         p[25]=tau, p[26:28]=sigma, p[29:37]=adj_flat, p[38:40]=active
# ==============================================================
# ODE function — deterministic part
# ==============================================================
function model_general_add(du, u, p, t)
    local_A = reshape(@view(p[29:37]), 3, 3)
   
    b = @view p[1:3]
    d = @view p[4:6]
    τ = p[25]
    
    active = @view p[38:40] # 1.0 = active node, 0.0 = inactive node
   
    fill!(du, 0.0)
    for i in 1:N_SPECIES
        reg = 1.0
        for j in 1:N_SPECIES
            if local_A[i,j] != 0.0
                link_index = (i-1)*N_SPECIES + j
                K = p[6 + link_index]
                n = p[15 + link_index]
                
                if local_A[i,j] > 0
                    reg *= hill_act(u[j], K, n, V_CONST)
                else
                    reg *= hill_inh(u[j], K, n, V_CONST)
                end
            end
        end
       
        # Only compute du[i] if the node is active
        if active[i] == 1.0
            du[i] = b[i] * V_CONST + reg * V_CONST - d[i] * u[i] + u[N_SPECIES + i]
        else
            du[i] = 0.0 # inactive → no change
        end
    end
   
    # OU part (mean reversion) — only for active nodes
    for i in 1:N_OU
        if active[i] == 1.0
            du[N_SPECIES + i] = -u[N_SPECIES + i] / τ
        else
            du[N_SPECIES + i] = 0.0 # inactive → no noise dynamics
        end
    end
end
# ==============================================================
# Noise function
# ==============================================================
function stoch_add(du, u, p, t)    
    τ      = p[25]
    σ      = @view p[26:28]
    active = @view p[38:40]
   
    for i in 1:N_SPECIES
        du[i] = 0.0
    end
   
    for i in 1:N_OU
        if active[i] == 1.0
            du[N_SPECIES + i] = sqrt(2.0 * σ[i]^2 / τ)
        else
            du[N_SPECIES + i] = 0.0 # no noise for inactive nodes
        end
    end
end
# ==============================================================
# Main part
# ==============================================================
Random.seed!(2025)
if isfile("unique_networks_table.jld2")
    @load "unique_networks_table.jld2" network_table
    global nets = [(id=n.unique_id, adj_matrix=n.adj_matrix) for n in network_table]
    logmsg("Loaded unique_networks_table.jld2 → found $(length(network_table)) network(s)")
else
    @warn "unique_networks_table.jld2 not found → using placeholder network 8"
    global nets = [(id=8, adj_matrix=[0 0 -1; -1 0 0; 0 -1 0])]
    logmsg("Using placeholder network id=8")
end
start_time = now()

# Select only a few specific networks by their unique_id

const DESIRED_NET_IDS = 1:100

nets_to_run = filter(n -> n.id in DESIRED_NET_IDS, nets)

if isempty(nets_to_run)
    error("None of the desired network IDs were found in the loaded data.")
end

logmsg("Processing only $(length(nets_to_run)) selected networks: $([n.id for n in nets_to_run])")

# File names updated for selected/few networks
file = "multimodal_search_selected_1.h5"
csv_file = "multimodal_results_selected_1.csv"

logmsg("STARTED — Selected networks processing | $(length(nets_to_run)) networks")
const times_save = 0.0:50.0:2000.0
const times_full = 0.0:1.0:2000.0
const tspan = (0.0, 2000.0)
const τ0 = 0.1
const σ0 = 1.0 # change the noise level as needed
const FIXED_PARAMS = (τ0, σ0, σ0, σ0)
param = generate_lognormal_params(Np, PARAM_RANGES_GENERAL)
h5open(file, "w") do f
    f["times_save"] = collect(times_save)
    f["tau"] = τ0
    f["sigma"] = σ0
    for net in nets_to_run
        id = net.id
        adj = net.adj_matrix
        active_mask = [any(adj[i,:] .!= 0) || any(adj[:,i] .!= 0) for i in 1:3]
        active_indices = findall(active_mask)
        n_active = length(active_indices)
        logmsg("=== Processing net_$id (Active nodes: $n_active → indices $active_indices) ===")
        adj_flat = vec(adj)
        active_mask_float = Float64.(active_mask)
        g_net = create_group(f, "net_$id")
        g_net["adj_matrix"] = adj
        net_seed_base = BASE_SEED_CONSTANT + (id * NETWORK_SEED_MULTIPLIER)
       
        for p_idx in 1:Np
            sampled_params = param[p_idx, 1:N_SAMPLED_PARAMS]
            fullp = vcat(sampled_params, FIXED_PARAMS..., adj_flat, active_mask_float)
           
            # --- DETERMINISTIC PRE-RUN FOR INITIAL CONDITIONS ---
            # 1. Automatic IC based on first active node
           
            u0_det_species = zeros(3)
            if !isempty(active_indices)
                u0_det_species[active_indices[1]] = 1.0
            end
           
            u0_deterministic = vcat(u0_det_species, zeros(N_OU))
           
            # 2. Solve using CVODE
            prob_ode = ODEProblem(model_general_add, u0_deterministic, tspan, fullp)
            sol_ode = solve(prob_ode, CVODE_BDF(), reltol=1e-6, abstol=1e-6, saveat=times_save)
          
            # 3. Capture species values at t=1500
            u_captured = sol_ode(1500.0)[1:3]
           
            # 4. Use deterministic IC at t=1500 for the single IC
            u0_local = zeros(Nic, 3)
            if !isempty(active_indices)
                for actual_idx in active_indices
                    u0_local[1, actual_idx] = max(u_captured[actual_idx], MINVAL)
                end
            end
           
            g_param = create_group(g_net, "param_$p_idx")
            #g_param["ode_trajectory"] = reduce(hcat, sol_ode.u)[1:3, :]
            temp_ode_traj = reduce(hcat, sol_ode.u)[1:3, :]
            g_param["b"] = sampled_params[1:3]
            g_param["d"] = sampled_params[4:6]
            g_param["K_matrix_values"] = sampled_params[7:15]
            g_param["n_matrix_values"] = sampled_params[16:24]
            g_param["V"]               = V_CONST 
           
            param_seed_base = net_seed_base + (p_idx * PARAM_SEED_MULTIPLIER)
           
            saved_traj_data = Vector{Matrix{Float32}}(undef, Nic * Nt)
            full_traj_x1 = Vector{Vector{Float64}}(undef, Nic * Nt)
            full_traj_x2 = Vector{Vector{Float64}}(undef, Nic * Nt)
            full_traj_x3 = Vector{Vector{Float64}}(undef, Nic * Nt)
            final_states = zeros(Nic * Nt, 3)
           
            GC.enable(false)
           
            @threads for linear_idx in 1:(Nic * Nt)
                ic = div(linear_idx - 1, Nt) + 1
                tr = mod(linear_idx - 1, Nt) + 1
               
                # Use the locally generated LHS initial conditions
                species_u0 = u0_local[ic, :]
                u0full = vcat(species_u0, zeros(N_OU))
               
                trajectory_seed = param_seed_base + ((ic - 1) * Nt) + (tr - 1)
                Random.seed!(trajectory_seed)
               
                prob = SDEProblem(model_general_add, stoch_add, copy(u0full), tspan, fullp)
                sol = solve(prob, SROCK1(); saveat=times_save, dt=0.1, maxiters=1e7, abstol=1e-6, reltol=1e-4)
               
                coarse_3sp = reduce(hcat, sol.u)[1:N_SPECIES, :]
                coarse_3sp[coarse_3sp .< 0.0] .= MINVAL
                saved_traj_data[linear_idx] = Float32.(coarse_3sp)
                final_states[linear_idx, :] = coarse_3sp[:, end]
                full_sol = sol(times_full)
                full_3sp = reduce(hcat, full_sol.u)[1:N_SPECIES, :]
               
                full_3sp[full_3sp .< 0.0] .= MINVAL
                full_traj_x1[linear_idx] = vec(full_3sp[1, :])
                full_traj_x2[linear_idx] = vec(full_3sp[2, :])
                full_traj_x3[linear_idx] = vec(full_3sp[3, :])
            end
           
            GC.enable(true)
            GC.gc()
           
            # ────────────────────────────────────────────────────────────────
            # SINGLE NODE CASE → 1D GMM
            # ────────────────────────────────────────────────────────────────
            if n_active == 1
                active_species = active_indices[1]

                if active_species == 1
                    data_1d = final_states[:, 1]
                    traj_1d = full_traj_x1
                elseif active_species == 2
                    data_1d = final_states[:, 2]
                    traj_1d = full_traj_x2
                else
                    data_1d = final_states[:, 3]
                    traj_1d = full_traj_x3
                end

                function compute_threshold_1d(data_vec)
                    lock(py_lock) do
                        try
                            d = reshape(data_vec, :, 1)
                            gmm1 = sklearn.GaussianMixture(n_components=1, covariance_type="full", random_state=0).fit(d)
                            gmm2 = sklearn.GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(d)
                            if gmm2.bic(d) < gmm1.bic(d)
                                labels = gmm2.predict(d)
                                c0 = d[labels .== 0, 1]
                                c1 = d[labels .== 1, 1]
                                if length(c0) < 10 || length(c1) < 10
                                    return false, NaN, nothing
                                end
                                m0 = mean(c0)
                                m1 = mean(c1)
                                thresh = (m0 + m1) / 2
                                sep_required = 0.2 * max(abs(m0), abs(m1))
                                if abs(m0 - m1) < sep_required
                                    return false, NaN, nothing
                                end
                                return true, thresh, gmm2
                            end
                        catch e
                        end
                        return false, NaN, nothing
                    end
                end

                ok_1d, thresh_1d, ~ = compute_threshold_1d(data_1d)

                if !ok_1d
                    attrs(g_param)["multimodal"] = false
                    continue
                end

                threshold_x1 = active_species == 1 ? thresh_1d : NaN
                threshold_x2 = active_species == 2 ? thresh_1d : NaN
                threshold_x3 = active_species == 3 ? thresh_1d : NaN

                on_fraction_active = Float64[]
                off_fraction_active = Float64[]
                total_on = total_off = total_pop = 0.0

                for ti in 1:length(times_full)
                    vals = [traj_1d[i][ti] for i in 1:Nic*Nt]
                    on_cnt = count(>(thresh_1d), vals)
                    off_cnt = length(vals) - on_cnt
                    total_pop += length(vals)
                    total_on += on_cnt
                    total_off += off_cnt
                    push!(on_fraction_active, total_on / total_pop)
                    push!(off_fraction_active, total_off / total_pop)
                end

                final_on_frac = on_fraction_active[end]

                stability_time_point = NaN
                tracking = false
                for (ti, t) in enumerate(times_full)
                    cur_on = on_fraction_active[ti]
                    if 0.98*final_on_frac <= cur_on <= 1.02*final_on_frac
                        if !tracking
                            tracking = true
                            stability_time_point = t
                        end
                    else
                        tracking = false
                    end
                end

                cv_before = [NaN, NaN, NaN]
                cv_after = [NaN, NaN, NaN]

                if !isnan(stability_time_point)
                    stab_idx = findfirst(==(stability_time_point), times_full)
                    stab_idx = isnothing(stab_idx) ? length(times_full) : stab_idx

                    seg_before = [traj_1d[j][1:(stab_idx-1)] for j in 1:Nic*Nt if stab_idx > 1]
                    seg_after  = [traj_1d[j][stab_idx:end] for j in 1:Nic*Nt]

                    function safe_cv(vals)
                        m = mean(vals)
                        m > 1e-8 ? std(vals)/m : NaN
                    end

                    cv_before_active = mean(skipmissing(safe_cv(s) for s in seg_before))
                    cv_after_active  = mean(skipmissing(safe_cv(s) for s in seg_after))

                    cv_before[active_species] = cv_before_active
                    cv_after[active_species]  = cv_after_active
                end

                # compute path length on the fly 
                path_lengths_1d = Float64[]
                let stab_idx_path = if isnan(stability_time_point)
                                        length(times_full)
                                    else
                                        something(findfirst(==(stability_time_point), times_full),
                                                  length(times_full))
                                    end
                    for tr_i in 1:Nic*Nt
                        traj = traj_1d[tr_i]
                        limit = min(length(traj), stab_idx_path)
                        if limit > 1
                            push!(path_lengths_1d, sum(abs.(diff(traj[1:limit]))))
                        else
                            push!(path_lengths_1d, 0.0)
                        end
                    end
                end
                mean_path_1d   = mean(path_lengths_1d)
                std_path_1d    = std(path_lengths_1d)
                median_path_1d = median(path_lengths_1d)

                attrs(g_param)["multimodal"] = true
                attrs(g_param)["threshold_x1"] = threshold_x1
                attrs(g_param)["threshold_x2"] = threshold_x2
                attrs(g_param)["threshold_x3"] = threshold_x3
                attrs(g_param)["stability_time"] = stability_time_point
                attrs(g_param)["cv_before_x1"] = cv_before[1]
                attrs(g_param)["cv_before_x2"] = cv_before[2]
                attrs(g_param)["cv_before_x3"] = cv_before[3]
                attrs(g_param)["cv_after_x1"] = cv_after[1]
                attrs(g_param)["cv_after_x2"] = cv_after[2]
                attrs(g_param)["cv_after_x3"] = cv_after[3]

                
                push!(summary_df, (
                id, p_idx,
                threshold_x1, threshold_x2, threshold_x3,
                stability_time_point,
                cv_before[1], cv_before[2], cv_before[3],
                cv_after[1], cv_after[2], cv_after[3],
                mean_path_1d, std_path_1d, median_path_1d
            ))

                logmsg("net_$id param_$p_idx → SINGLE-NODE MULTIMODAL | stab_time = $stability_time_point")
                continue
            end

            # Analysis functions            
            function compute_threshold(data)
                lock(py_lock) do
                    try
                        d_mat = Matrix(data)
                        gmm1 = sklearn.GaussianMixture(n_components=1, covariance_type="full",
                                   random_state=0).fit(d_mat)
                        gmm2 = sklearn.GaussianMixture(n_components=2, covariance_type="full",
                                   random_state=0).fit(d_mat)
                        if gmm2.bic(d_mat) < gmm1.bic(d_mat)
                            labels = gmm2.predict(d_mat)
                            c0 = d_mat[labels .== 0, :]
                            c1 = d_mat[labels .== 1, :]
                            if isempty(c0) || isempty(c1)      
                                return false, [NaN, NaN], nothing  
                            end                                    
                            m0 = mean(c0, dims=1)[:]           
                            m1 = mean(c1, dims=1)[:]
                            return true, (m0 .+ m1) ./ 2, gmm2
                        end                                   
                    catch e
                    end
                    return false, [NaN, NaN], nothing
                end
            end            
                       
            function well_separated(m1, m2)
                all(abs(m1[k] - m2[k]) >= 0.2 * max(abs(m1[k]), abs(m2[k])) for k in 1:2)
            end
            data12 = final_states[:, 1:2]
            data13 = final_states[:, [1, 3]]
            data23 = final_states[:, 2:3]
           
            ok12, th12, gmm12_opt = compute_threshold(data12)
            ok13, th13, gmm13_opt = compute_threshold(data13)
            ok23, th23, gmm23_opt = compute_threshold(data23)
           
            if !(ok12 && ok13 && ok23)
                attrs(g_param)["multimodal"] = false
                #logmsg("net_$id param_$p_idx → discarded (BIC failed)")
                continue
            end
                      
            lbl12 = gmm12_opt.predict(data12)
            m12_1 = mean(data12[lbl12 .== 0, :], dims=1)[:]
            m12_2 = mean(data12[lbl12 .== 1, :], dims=1)[:]
           
            lbl13 = gmm13_opt.predict(data13)
            m13_1 = mean(data13[lbl13 .== 0, :], dims=1)[:]
            m13_2 = mean(data13[lbl13 .== 1, :], dims=1)[:]
            lbl23 = gmm23_opt.predict(data23)
            m23_1 = mean(data23[lbl23 .== 0, :], dims=1)[:]
            m23_2 = mean(data23[lbl23 .== 1, :], dims=1)[:]
           
           
            if !(well_separated(m12_1, m12_2) && well_separated(m13_1, m13_2) && well_separated(m23_1, m23_2))
                attrs(g_param)["multimodal"] = false
                #logmsg("net_$id param_$p_idx → discarded (separation failed)")
                continue
            end
          
            # Save trajectory data for multimodal sets
            idx = 1
            for ic_idx in 1:Nic
                g_ic = create_group(g_param, "ic_$ic_idx")
                g_ic["u0"] = u0_local[ic_idx, :]
                for tr in 1:Nt
                    g_ic["trial_$tr"] = saved_traj_data[idx]
                    idx += 1
                end
            end
            threshold_x1 = (th12[1] + th13[1]) / 2
            threshold_x2 = (th12[2] + th23[1]) / 2
            threshold_x3 = (th13[2] + th23[2]) / 2
           
            attrs(g_param)["multimodal"] = true
            g_param["ode_trajectory"] = temp_ode_traj # ODE data saved for only multimodal sets
           
            attrs(g_param)["threshold_x1"] = threshold_x1
            attrs(g_param)["threshold_x2"] = threshold_x2
            attrs(g_param)["threshold_x3"] = threshold_x3
           
            on_fraction_x1 = Float64[]; off_fraction_x1 = Float64[]
            on_fraction_x2 = Float64[]; off_fraction_x2 = Float64[]
            on_fraction_x3 = Float64[]; off_fraction_x3 = Float64[]
           
            total_on_x1 = total_off_x1 = total_on_x2 = total_off_x2 = total_on_x3 = total_off_x3 = 0.0
            total_population_x1 = total_population_x2 = total_population_x3 = 0.0
           
            for ti in 1:length(times_full)
                x1_vals = [full_traj_x1[i][ti] for i in 1:Nic*Nt]
                x2_vals = [full_traj_x2[i][ti] for i in 1:Nic*Nt]
                x3_vals = [full_traj_x3[i][ti] for i in 1:Nic*Nt]
               
                on_x1 = count(>(threshold_x1), x1_vals); off_x1 = length(x1_vals) - on_x1
                on_x2 = count(>(threshold_x2), x2_vals); off_x2 = length(x2_vals) - on_x2
                on_x3 = count(>(threshold_x3), x3_vals); off_x3 = length(x3_vals) - on_x3
               
                total_population_x1 += length(x1_vals)
                total_population_x2 += length(x2_vals)
                total_population_x3 += length(x3_vals)
               
                total_on_x1 += on_x1; total_off_x1 += off_x1
                total_on_x2 += on_x2; total_off_x2 += off_x2
                total_on_x3 += on_x3; total_off_x3 += off_x3
               
                push!(on_fraction_x1, total_on_x1 / total_population_x1)
                push!(off_fraction_x1, total_off_x1 / total_population_x1)
                push!(on_fraction_x2, total_on_x2 / total_population_x2)
                push!(off_fraction_x2, total_off_x2 / total_population_x2)
                push!(on_fraction_x3, total_on_x3 / total_population_x3)
                push!(off_fraction_x3, total_off_x3 / total_population_x3)
            end
           
            final_on = [on_fraction_x1[end], on_fraction_x2[end], on_fraction_x3[end]]
           
            stability_time_point = NaN
            tracking = false
            for (ti, t) in enumerate(times_full)
                cur_on = [on_fraction_x1[ti], on_fraction_x2[ti], on_fraction_x3[ti]]
                if all(k -> 0.98*final_on[k] <= cur_on[k] <= 1.02*final_on[k], 1:3)
                    if !tracking
                        tracking = true
                        stability_time_point = t
                    end
                else
                    tracking = false
                end
            end
            attrs(g_param)["stability_time"] = stability_time_point
            cv_before = [NaN, NaN, NaN]
            cv_after = [NaN, NaN, NaN]
           
            if !isnan(stability_time_point)
                stab_idx = findfirst(==(stability_time_point), times_full)
                stab_idx = isnothing(stab_idx) ? length(times_full) : stab_idx
                function avg_cv(trajs, a, b)
                    if a > b
                        return NaN
                    end
                    cvs = Float64[]
                    for traj in trajs
                        seg = traj[a:b]
                        if !isempty(seg) && mean(seg) > 1e-9
                            push!(cvs, std(seg) / mean(seg))
                        end
                    end
                    isempty(cvs) ? NaN : mean(cvs)
                end
                cv_before[1] = avg_cv(full_traj_x1, 1, stab_idx - 1)
                cv_before[2] = avg_cv(full_traj_x2, 1, stab_idx - 1)
                cv_before[3] = avg_cv(full_traj_x3, 1, stab_idx - 1)
                cv_after[1] = avg_cv(full_traj_x1, stab_idx, length(times_full))
                cv_after[2] = avg_cv(full_traj_x2, stab_idx, length(times_full))
                cv_after[3] = avg_cv(full_traj_x3, stab_idx, length(times_full))
            end
            
            # compute path length on the fly for multi-node multimodal sets
            path_lengths = Float64[]
            let stab_idx_path = if isnan(stability_time_point)
                                    length(times_full)
                                else
                                    something(findfirst(==(stability_time_point), times_full),
                                              length(times_full))
                                end
                for tr_i in 1:Nic*Nt
                    traj = full_traj_x1[tr_i]
                    limit = min(length(traj), stab_idx_path)
                    if limit > 1
                        push!(path_lengths, sum(abs.(diff(traj[1:limit]))))
                    else
                        push!(path_lengths, 0.0)
                    end
                end
            end
            mean_path_val   = mean(path_lengths)
            std_path_val    = std(path_lengths)
            median_path_val = median(path_lengths)
             
            
            attrs(g_param)["cv_before_x1"] = cv_before[1]
            attrs(g_param)["cv_before_x2"] = cv_before[2]
            attrs(g_param)["cv_before_x3"] = cv_before[3]
            attrs(g_param)["cv_after_x1"] = cv_after[1]
            attrs(g_param)["cv_after_x2"] = cv_after[2]
            attrs(g_param)["cv_after_x3"] = cv_after[3]
            logmsg("net_$id param_$p_idx → MULTIMODAL | stab_time = $stability_time_point")           
            push!(summary_df, (
                id, p_idx,
                threshold_x1, threshold_x2, threshold_x3,
                stability_time_point,
                cv_before[1], cv_before[2], cv_before[3],
                cv_after[1], cv_after[2], cv_after[3],
                mean_path_val, std_path_val, median_path_val
            ))
        end
        logmsg("-> net_$id done")
      
        # CLEAR MEMORY HERE
        GC.gc()
    end
end
CSV.write(csv_file, summary_df)
total_time = Dates.canonicalize(now() - start_time)
logmsg("FINISHED! Found $(nrow(summary_df)) multimodal parameter sets.")
logmsg("Results saved to: $csv_file")
logmsg("Total runtime: $total_time")