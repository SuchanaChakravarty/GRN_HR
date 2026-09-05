using PyCall
const py_lock = ReentrantLock()
py"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
"""
using JLD2, DifferentialEquations, Random, HDF5, Dates, Sundials
using LinearAlgebra, Statistics, DataFrames, CSV, DelimitedFiles
using LatinHypercubeSampling
using Base.Threads
sklearn = pyimport("sklearn.mixture")
# ================================================================================
# Mode switch: default (ODE-derived) Initial Ccondition (IC) vs precomputed IC
#
#   julia run_selected_net.jl              -> default IC (all Np params/net)
#   julia run_selected_net.jl precomputed  -> IC read from ic_from_far.dat,
#                                              only the (net,param) pairs
#                                              listed there are run
# =================================================================================
const USE_PRECOMPUTED_IC = !isempty(ARGS) && lowercase(ARGS[1]) in ("precomputed", "precomputed_ic", "true")
const SUFFIX = USE_PRECOMPUTED_IC ? "_op_ic" : ""
# ===================================================================================
# Progress logging
# ==============================================================
const LOG_FILE = "progress_new_1$(SUFFIX).log"
function logmsg(msg)
    ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    open(LOG_FILE, "a") do f
        println(f, "[$ts] [T$(threadid())] $msg")
    end
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
# Species names
# ==============================================================
const SPECIES = ["x1", "x2", "x3"]
# ==============================================================
# Summary table
# ==============================================================
summary_df = DataFrame(
    net_id = Int[],
    param_id = Int[],
    accepted_nodes = String[], discarded_node = String[], threshold_equiv = Float32[],
    ic_x1 = Float32[], ic_x2 = Float32[], ic_x3 = Float32[],
    gmm_mean1_x1 = Float32[], gmm_mean2_x1 = Float32[],
    gmm_mean1_x2 = Float32[], gmm_mean2_x2 = Float32[],
    gmm_mean1_x3 = Float32[], gmm_mean2_x3 = Float32[],
    far_from_ic_x1 = Float32[], far_from_ic_x2 = Float32[], far_from_ic_x3 = Float32[],
    overshoot_ratio = Float32[], overshoot_time = Float32[], overshoot_state = String[],
    stability_time = Float32[],
    cv_before_x1 = Float32[], cv_before_x2 = Float32[], cv_before_x3 = Float32[],
    cv_after_x1 = Float32[], cv_after_x2 = Float32[], cv_after_x3 = Float32[],
    mean_path_x1 = Float32[], std_path_x1 = Float32[], median_path_x1 = Float32[],
    mean_path_x2 = Float32[], std_path_x2 = Float32[], median_path_x2 = Float32[],
    mean_path_x3 = Float32[], std_path_x3 = Float32[], median_path_x3 = Float32[]
)
# ==============================================================
# Overshoot table
# ==============================================================
overshoot_df = DataFrame(
    net_id = Int[],
    param_id = Int[],
    accepted_nodes = String[],
    on_final = Float32[], on_peak_val = Float32[], on_peak_time = Float32[],
    on_which = String[], on_ratio = Float32[],
    off_final = Float32[], off_peak_val = Float32[], off_peak_time = Float32[],
    off_which = String[], off_ratio = Float32[]
)
# ==============================================================
# Parameter ranges
# ==============================================================
const PARAM_RANGES_GENERAL = vcat(
    fill((0.001, 0.1),  3),   # b
    fill((0.01,  0.5), 3),   # d
    fill((0.1,   5.0),  9),   # K
    fill((1.0,   10.0), 9)    # n
)

const V_CONST = 1.0

const N_SAMPLED_PARAMS = length(PARAM_RANGES_GENERAL)
const Np, Nic, Nt = 1000, 1, 500

using SpecialFunctions

function normal_cdf(x::Float64)
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))
end

function normal_invcdf(p::Float64)
    return sqrt(2.0) * erfinv(2.0 * p - 1.0)
end

function sample_truncated_lognormal(a::Float64, b::Float64, mu::Float64, sigma::Float64)
    p_low  = normal_cdf((log(a) - mu) / sigma)
    p_high = normal_cdf((log(b) - mu) / sigma)
    u = p_low + rand() * (p_high - p_low)
    u = clamp(u, 1e-14, 1.0 - 1e-14)
    return exp(mu + sigma * normal_invcdf(u))
end

function generate_lognormal_params(num_samples, ranges; max_attempts=1000)
    num_dims = length(ranges)
    for _ in 1:max_attempts
        samples = zeros(num_samples, num_dims)
        for (j, (a, b)) in enumerate(ranges)
            mu    = (log(a) + log(b)) / 2.0
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
    (m^n) / (K^n + m^n)
end
@inline hill_inh(x, K, n, V) = let m = max(x / V, HILL_EPS)
    K^n / (K^n + m^n)
end

# Layout: p[1:3]=b, p[4:6]=d, p[7:15]=K, p[16:24]=n,
#         p[25]=tau, p[26:28]=sigma, p[29:37]=adj_flat, p[38:40]=active
function model_general_add(du, u, p, t)
    local_A = reshape(@view(p[29:37]), 3, 3)

    b = @view p[1:3]
    d = @view p[4:6]
    τ = p[25]

    active = @view p[38:40]

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

        if active[i] == 1.0
            du[i] = b[i] * V_CONST + reg * V_CONST - d[i] * u[i] + u[N_SPECIES + i]
        else
            du[i] = 0.0
        end
    end

    for i in 1:N_OU
        if active[i] == 1.0
            du[N_SPECIES + i] = -u[N_SPECIES + i] / τ
        else
            du[N_SPECIES + i] = 0.0
        end
    end
end

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
            du[N_SPECIES + i] = 0.0
        end
    end
end
# ==============================================================
# Analysis helper functions
# ==============================================================
function settling_time(t::AbstractVector{<:Real}, x::AbstractVector{<:Real};
                        tail_time::Real=200,
                        band::Real=0.05)
    n = length(x)
    (n == length(t) && n > 0) || return NaN

    t_start = t[end] - tail_time
    steady_idx = findall(t .>= t_start)
    isempty(steady_idx) && return NaN
    xf = mean(x[steady_idx])

    tol = abs(band)

    last_violation = 0
    @inbounds for i in n:-1:1
        if abs(x[i] - xf) > tol
            last_violation = i
            break
        end
    end

    last_violation == 0 && return t[1]
    last_violation == n && return NaN
    return t[last_violation + 1]
end

function far_from_ic(mean_low, mean_high, ic_val)
    if isnan(mean_low) || isnan(mean_high) || isnan(ic_val)
        return NaN
    end
    dist_low  = abs(mean_low  - ic_val)
    dist_high = abs(mean_high - ic_val)
    return dist_high >= dist_low ? mean_high : mean_low
end

function gated_peak(frac::Vector{Float64}, times::Vector{Float64}, final_val::Float64)
    final_val < 1e-10 && return NaN, NaN, "none", NaN

    initial_val = frac[1]
    thresh = max(final_val * (1.0 + TOL), initial_val + 1e-5)

    idx1 = something(findlast(t -> t <= EARLY_CUTOFF_1, times), 0)
    if idx1 >= 1
        v1, i1 = findmax(frac[1:idx1])
        v1 > thresh && return v1, times[i1], "early", v1 / final_val
    end

    idx2_start = something(findfirst(t -> t > EARLY_CUTOFF_1, times), idx1 + 1)
    idx2_end   = something(findlast(t -> t <= EARLY_CUTOFF_2, times), idx2_start - 1)
    if idx2_end >= idx2_start
        v2, rel = findmax(frac[idx2_start:idx2_end])
        i2 = idx2_start + rel - 1
        v2 > thresh && return v2, times[i2], "late", v2 / final_val
    end

    return NaN, NaN, "none", NaN
end

function primary_overshoot(on_ratio, on_pt, on_which, off_ratio, off_pt, off_which)
    if on_which != "none"
        return on_ratio, on_pt, "on"
    elseif off_which != "none"
        return off_ratio, off_pt, "off"
    else
        return NaN, NaN, "none"
    end
end

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
# ==============================================================
# GMM posterior-density functions
# ==============================================================
function fit_gmm_1d(data_vec::Vector{Float64})
    lock(py_lock) do
        try
            d = reshape(data_vec, :, 1)
            gmm1 = sklearn.GaussianMixture(n_components=1, covariance_type="full", random_state=0).fit(d)
            gmm2 = sklearn.GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(d)
            bic1, bic2 = gmm1.bic(d), gmm2.bic(d)
            if bic2 < bic1
                labels = gmm2.predict(d)
                c0 = d[labels .== 0, 1]; c1 = d[labels .== 1, 1]
                if length(c0) < 10 || length(c1) < 10
                    return (ok=false, gmm=nothing, bic=bic2, frac0=NaN, frac1=NaN)
                end
                frac0, frac1 = length(c0)/length(data_vec), length(c1)/length(data_vec)
                if frac0 < MIN_BALANCE_FRAC || frac1 < MIN_BALANCE_FRAC
                    return (ok=false, gmm=nothing, bic=bic2, frac0=NaN, frac1=NaN)
                end
                m0, m1 = mean(c0), mean(c1)
                sep_required = 0.3 * max(abs(m0), abs(m1))
                if abs(m0 - m1) < sep_required
                    return (ok=false, gmm=nothing, bic=bic2, frac0=NaN, frac1=NaN)
                end
                return (ok=true, gmm=gmm2, bic=bic2, frac0=frac0, frac1=frac1)
            end
        catch e
        end
        return (ok=false, gmm=nothing, bic=NaN, frac0=NaN, frac1=NaN)
    end
end

function fit_gmm_2d(data_mat)
    lock(py_lock) do
        try
            d = Matrix(data_mat)
            gmm1 = sklearn.GaussianMixture(n_components=1, covariance_type="full", random_state=0).fit(d)
            gmm2 = sklearn.GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(d)
            bic1, bic2 = gmm1.bic(d), gmm2.bic(d)
            bic_pass = bic2 < bic1
            if bic_pass
                labels = gmm2.predict(d)
                c0 = d[labels .== 0, :]; c1 = d[labels .== 1, :]
                if isempty(c0) || isempty(c1)
                    return (stage1_pass=false, balance_pass=false, gmm=nothing, bic=bic2, frac0=NaN, frac1=NaN)
                end
                m0 = mean(c0, dims=1)[:]; m1 = mean(c1, dims=1)[:]
                well_sep = all(abs(m0[k]-m1[k]) >= 0.3*max(abs(m0[k]),abs(m1[k])) for k in 1:2)
                if !well_sep
                    return (stage1_pass=false, balance_pass=false, gmm=nothing, bic=bic2, frac0=NaN, frac1=NaN)
                end
                frac0, frac1 = size(c0,1)/size(d,1), size(c1,1)/size(d,1)
                balance_pass = frac0 >= MIN_BALANCE_FRAC && frac1 >= MIN_BALANCE_FRAC
                return (stage1_pass=true, balance_pass=balance_pass, gmm=gmm2, bic=bic2, frac0=frac0, frac1=frac1)
            else
                return (stage1_pass=false, balance_pass=false, gmm=nothing, bic=bic2, frac0=NaN, frac1=NaN)
            end
        catch e
        end
        return (stage1_pass=false, balance_pass=false, gmm=nothing, bic=NaN, frac0=NaN, frac1=NaN)
    end
end

function on_component_1d(gmm)
    means = vec(gmm.means_)
    return means[1] >= means[2] ? 0 : 1
end

function on_component_2d(gmm)
    means = gmm.means_
    s0 = sum(means[1, :]); s1 = sum(means[2, :])
    return s0 >= s1 ? 0 : 1
end

function classify_posterior_1d(gmm, on_comp::Int, vals::Vector{Float64})
    d = reshape(vals, :, 1)
    proba = gmm.predict_proba(d)
    return proba[:, on_comp+1] .>= 0.5
end

function classify_posterior_2d(gmm, on_comp::Int, points::Matrix{Float64})
    proba = gmm.predict_proba(points)
    return proba[:, on_comp+1] .>= 0.5
end

function find_1d_threshold(gmm, on_comp::Int, data_range)
    xr = collect(range(minimum(data_range), maximum(data_range), length=2000))
    proba = gmm.predict_proba(reshape(xr, :, 1))[:, on_comp+1]
    for i in 1:(length(xr)-1)
        if (proba[i] - 0.5) * (proba[i+1] - 0.5) < 0
            frac = (0.5 - proba[i]) / (proba[i+1] - proba[i])
            return xr[i] + frac * (xr[i+1] - xr[i])
        end
    end
    return NaN
end
# ==============================================================
# Disjoint / null network detection
# ==============================================================
function active_node_indices(mat)
    active_mask = [any(mat[i,:] .!= 0) || any(mat[:,i] .!= 0) for i in 1:3]
    return findall(active_mask)
end

function find_components(active_indices::Vector{Int}, mat)
    parent = Dict(idx => idx for idx in active_indices)

    function find_root(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end

    function union_nodes!(x, y)
        rx, ry = find_root(x), find_root(y)
        if rx != ry
            parent[rx] = ry
        end
    end

    for i in active_indices, j in active_indices
        if i < j && (mat[i,j] != 0 || mat[j,i] != 0)
            union_nodes!(i, j)
        end
    end

    comps = Dict{Int,Vector{Int}}()
    for idx in active_indices
        r = find_root(idx)
        push!(get!(comps, r, Int[]), idx)
    end

    return collect(values(comps))
end

function is_null_or_disjoint(mat)
    all(mat .== 0) && return true
    active_idx = active_node_indices(mat)
    isempty(active_idx) && return true
    comps = find_components(active_idx, mat)
    return length(comps) > 1
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

n_before_filter = length(nets)
nets = filter(n -> !is_null_or_disjoint(n.adj_matrix), nets)
logmsg("Removed null/disjoint networks: $(n_before_filter) → $(length(nets)) remaining")

const DESIRED_NET_IDS = 1:200

# ==============================================================
# Precomputed-IC lookup (only loaded/used when USE_PRECOMPUTED_IC = true)
# File format: net_id  param_id  ic_x1  ic_x2  ic_x3   (NaN for inactive species)
# ==============================================================
const IC_LOOKUP = Dict{Tuple{Int,Int}, Vector{Float64}}()
const PARAM_IDS_BY_NET = Dict{Int, Vector{Int}}()
if USE_PRECOMPUTED_IC
    ic_dat_path = "ic_from_far.dat"
    _raw = readdlm(ic_dat_path, ' ', Float64; comments=true, comment_char='#')
    for r in 1:size(_raw, 1)
        n = Int(_raw[r, 1]);  p = Int(_raw[r, 2])
        IC_LOOKUP[(n, p)] = Float64[_raw[r, 3], _raw[r, 4], _raw[r, 5]]
        push!(get!(PARAM_IDS_BY_NET, n, Int[]), p)
    end
    for v in values(PARAM_IDS_BY_NET); sort!(unique!(v)); end
    logmsg("Loaded $(length(PARAM_IDS_BY_NET)) net_ids / $(length(IC_LOOKUP)) (net,param) ICs from $ic_dat_path")
end

nets_to_run = filter(n -> n.id in DESIRED_NET_IDS && (!USE_PRECOMPUTED_IC || n.id in keys(PARAM_IDS_BY_NET)), nets)

if isempty(nets_to_run)
    error("None of the desired network IDs were found in the loaded data.")
end

logmsg("Processing only $(length(nets_to_run)) selected networks: $([n.id for n in nets_to_run])")

file = "multimodal_search_net_1$(SUFFIX).h5"
frac_file = "fraction_data_net_1$(SUFFIX).h5"
csv_file = "multimodal_results_net_1$(SUFFIX).csv"
overshoot_csv_file = "overshoot_results_net_1$(SUFFIX).csv"

logmsg("STARTED — Selected networks processing | $(length(nets_to_run)) networks")
const times_save = 0.0:50.0:2000.0
const times_full = 0.0:1.0:2000.0
const times_full_vec = collect(Float64.(times_full))
const coarse_indices_in_full = [findfirst(==(t), times_full_vec) for t in times_save]
const tspan = (0.0, 2000.0)
const τ0 = 0.1
const σ0 = 1.0
const FIXED_PARAMS = (τ0, σ0, σ0, σ0)
const EARLY_CUTOFF_1 = 1000.0
const EARLY_CUTOFF_2 = 2000.0
const TOL            = 0.10
const MIN_BALANCE_FRAC = 0.05
param = generate_lognormal_params(Np, PARAM_RANGES_GENERAL)
h5open(file, "w") do f
h5open(frac_file, "w") do f2
    f["times_save"] = Float32.(collect(times_save))
    f["times_full"] = Float32.(times_full_vec)
    f["tau"] = Float32(τ0)
    f["sigma"] = Float32(σ0)
    f2["times_save"] = Float32.(collect(times_save))
    f2["sigma"] = Float32(σ0)
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
        g_net2 = create_group(f2, "net_$id")
        g_net["adj_matrix"] = adj
        net_seed_base = BASE_SEED_CONSTANT + (id * NETWORK_SEED_MULTIPLIER)

        param_ids_this_net = USE_PRECOMPUTED_IC ? PARAM_IDS_BY_NET[id] : (1:Np)
        for p_idx in param_ids_this_net
          try
            sampled_params = param[p_idx, 1:N_SAMPLED_PARAMS]
            fullp = vcat(sampled_params, FIXED_PARAMS..., adj_flat, active_mask_float)

            u0_det_species = zeros(3)
            if !isempty(active_indices)
                u0_det_species[active_indices[1]] = 1.0
            end

            u0_deterministic = vcat(u0_det_species, zeros(N_OU))

            prob_ode = ODEProblem(model_general_add, u0_deterministic, tspan, fullp)
            sol_ode = solve(prob_ode, CVODE_BDF(), reltol=1e-6, abstol=1e-6, saveat=times_save)

            u0_local = zeros(Nic, 3)
            if USE_PRECOMPUTED_IC
                ic_csv = IC_LOOKUP[(id, p_idx)]
                for actual_idx in active_indices
                    v = ic_csv[actual_idx]
                    u0_local[1, actual_idx] = isnan(v) ? MINVAL : max(v, MINVAL)
                end
            else
                u_captured = sol_ode(1500.0)[1:3]
                for actual_idx in active_indices
                    u0_local[1, actual_idx] = max(u_captured[actual_idx], MINVAL)
                end
            end

            g_param = create_group(g_net, "param_$p_idx")
            g_param2 = create_group(g_net2, "param_$p_idx")
            temp_ode_traj = reduce(hcat, sol_ode.u)[1:3, :]
            g_param["b"] = Float32.(sampled_params[1:3])
            g_param["d"] = Float32.(sampled_params[4:6])
            g_param["K_matrix_values"] = Float32.(sampled_params[7:15])
            g_param["n_matrix_values"] = Float32.(sampled_params[16:24])
            g_param["V"]               = Float32(V_CONST)

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

                species_u0 = u0_local[ic, :]
                u0full = vcat(species_u0, zeros(N_OU))

                trajectory_seed = param_seed_base + ((ic - 1) * Nt) + (tr - 1)
                Random.seed!(trajectory_seed)

                prob = SDEProblem(model_general_add, stoch_add, copy(u0full), tspan, fullp)
                sol = solve(prob, SROCK1(); saveat=times_save, dt=0.1, maxiters=1e7, abstol=1e-6, reltol=1e-4)

                coarse_3sp = reduce(hcat, sol.u)[1:N_SPECIES, :]
                coarse_3sp[coarse_3sp .< 0.0] .= MINVAL
                saved_traj_data[linear_idx] = Float32.(coarse_3sp)

                full_sol = sol(times_full)
                full_3sp = reduce(hcat, full_sol.u)[1:N_SPECIES, :]

                full_3sp[full_3sp .< 0.0] .= MINVAL
                final_states[linear_idx, :] = full_3sp[:, end]
                full_traj_x1[linear_idx] = vec(full_3sp[1, :])
                full_traj_x2[linear_idx] = vec(full_3sp[2, :])
                full_traj_x3[linear_idx] = vec(full_3sp[3, :])
            end

            GC.enable(true)
            GC.gc()

            traj_lookup = (full_traj_x1, full_traj_x2, full_traj_x3)

            fit = nothing
            on_comp = -1
            accepted_nodes = ""
            discarded_node = "none"
            threshold_equiv = NaN
            chosen_species = Int[]

            if n_active == 1
                sp = active_indices[1]
                res = fit_gmm_1d(final_states[:, sp])
                if !res.ok
                    attrs(g_param)["multimodal"] = false
                    continue
                end
                fit = res.gmm
                on_comp = on_component_1d(fit)
                accepted_nodes = "x$sp"
                chosen_species = [sp]
                threshold_equiv = find_1d_threshold(fit, on_comp, final_states[:, sp])

            elseif n_active == 2
                a, b = active_indices[1], active_indices[2]
                res = fit_gmm_2d(hcat(final_states[:, a], final_states[:, b]))
                if !(res.stage1_pass && res.balance_pass)
                    attrs(g_param)["multimodal"] = false
                    continue
                end
                fit = res.gmm
                on_comp = on_component_2d(fit)
                accepted_nodes = "x$(a),x$(b)"
                chosen_species = [a, b]

            else
                pairs = [(active_indices[1],active_indices[2]),
                         (active_indices[1],active_indices[3]),
                         (active_indices[2],active_indices[3])]
                fits = Dict{Tuple{Int,Int}, NamedTuple}()
                for (a,b) in pairs
                    fits[(a,b)] = fit_gmm_2d(hcat(final_states[:, a], final_states[:, b]))
                end
                if !all(v.stage1_pass for v in values(fits))
                    attrs(g_param)["multimodal"] = false
                    continue
                end
                balance_ok_pairs = [(k,v) for (k,v) in fits if v.balance_pass]
                if isempty(balance_ok_pairs)
                    attrs(g_param)["multimodal"] = false
                    continue
                end
                chosen_key = balance_ok_pairs[argmin([abs(v.frac0 - 0.5) for (k,v) in balance_ok_pairs])][1]
                res = fits[chosen_key]
                fit = res.gmm
                on_comp = on_component_2d(fit)
                a, b = chosen_key
                accepted_nodes = "x$(a),x$(b)"
                discarded_node = "x$(first(setdiff(Set(active_indices), Set((a,b)))))"
                chosen_species = [a, b]
            end

            attrs(g_param)["multimodal"] = true

            on_frac = zeros(Float64, length(times_full_vec))
            if n_active == 1
                sp = chosen_species[1]
                for ti in 1:length(times_full_vec)
                    vals_t = [traj_lookup[sp][i][ti] for i in 1:(Nic*Nt)]
                    on_frac[ti] = mean(classify_posterior_1d(fit, on_comp, vals_t))
                end
            else
                a, b = chosen_species[1], chosen_species[2]
                for ti in 1:length(times_full_vec)
                    va = [traj_lookup[a][i][ti] for i in 1:(Nic*Nt)]
                    vb = [traj_lookup[b][i][ti] for i in 1:(Nic*Nt)]
                    on_frac[ti] = mean(classify_posterior_2d(fit, on_comp, hcat(va, vb)))
                end
            end
            off_frac = 1.0 .- on_frac

            stability_time = settling_time(times_full_vec, on_frac, tail_time=200, band=0.05)
            attrs(g_param)["stability_time"] = Float32(stability_time)

            if isnan(stability_time)
                attrs(g_param)["stable"] = false
                logmsg("net_$id param_$p_idx → multimodal but never settles → rejected")
                continue
            end
            attrs(g_param)["stable"] = true

            idx_traj = 1
            for ic_idx in 1:Nic
                g_ic = create_group(g_param, "ic_$ic_idx")
                g_ic["u0"] = Float32.(u0_local[ic_idx, :])
                for tr in 1:Nt
                    g_ic["trial_$tr"] = saved_traj_data[idx_traj]
                    idx_traj += 1
                end
            end
            g_param["ode_trajectory"] = Float32.(temp_ode_traj)

            stab_idx = something(findfirst(==(stability_time), times_full_vec), length(times_full_vec))

            cv_before = [NaN, NaN, NaN]
            cv_after  = [NaN, NaN, NaN]
            mean_path_vec   = [NaN, NaN, NaN]
            std_path_vec    = [NaN, NaN, NaN]
            median_path_vec = [NaN, NaN, NaN]

            for k in active_indices
                trajs = traj_lookup[k]
                cv_before[k] = avg_cv(trajs, 1, stab_idx - 1)
                cv_after[k]  = avg_cv(trajs, stab_idx, length(times_full_vec))
                path_lengths = Float64[]
                for traj in trajs
                    limit = min(length(traj), stab_idx)
                    if limit > 1
                        push!(path_lengths, sum(abs.(diff(traj[1:limit]))))
                    else
                        push!(path_lengths, 0.0)
                    end
                end
                mean_path_vec[k]   = mean(path_lengths)
                std_path_vec[k]    = std(path_lengths)
                median_path_vec[k] = median(path_lengths)
            end

            gmm_mean1_vec = [NaN, NaN, NaN]
            gmm_mean2_vec = [NaN, NaN, NaN]
            off_comp = 1 - on_comp
            if n_active == 1
                sp = chosen_species[1]
                means = vec(fit.means_)
                gmm_mean1_vec[sp] = means[off_comp + 1]
                gmm_mean2_vec[sp] = means[on_comp + 1]
            else
                a, b = chosen_species[1], chosen_species[2]
                means = fit.means_
                gmm_mean1_vec[a] = means[off_comp + 1, 1]
                gmm_mean2_vec[a] = means[on_comp + 1, 1]
                gmm_mean1_vec[b] = means[off_comp + 1, 2]
                gmm_mean2_vec[b] = means[on_comp + 1, 2]
            end

            ic_vals = [u0_local[1, 1], u0_local[1, 2], u0_local[1, 3]]
            far_vec = [NaN, NaN, NaN]
            for k in chosen_species
                far_vec[k] = far_from_ic(gmm_mean1_vec[k], gmm_mean2_vec[k], ic_vals[k])
            end

            if discarded_node != "none"
                excluded_sp = parse(Int, replace(discarded_node, "x" => ""))
                pair_data = hcat(final_states[:, chosen_species[1]], final_states[:, chosen_species[2]])
                labels = fit.predict(pair_data)
                group_on  = final_states[labels .== on_comp, excluded_sp]
                group_off = final_states[labels .== off_comp, excluded_sp]
                if !isempty(group_on) && !isempty(group_off)
                    gmm_mean2_vec[excluded_sp] = mean(group_on)
                    gmm_mean1_vec[excluded_sp] = mean(group_off)
                    far_vec[excluded_sp] = far_from_ic(gmm_mean1_vec[excluded_sp], gmm_mean2_vec[excluded_sp], ic_vals[excluded_sp])
                end
            end

            attrs(g_param)["accepted_nodes"] = accepted_nodes
            attrs(g_param)["discarded_node"] = discarded_node
            attrs(g_param)["threshold_equiv"] = Float32(threshold_equiv)
            attrs(g_param)["cv_before_x1"] = Float32(cv_before[1])
            attrs(g_param)["cv_before_x2"] = Float32(cv_before[2])
            attrs(g_param)["cv_before_x3"] = Float32(cv_before[3])
            attrs(g_param)["cv_after_x1"] = Float32(cv_after[1])
            attrs(g_param)["cv_after_x2"] = Float32(cv_after[2])
            attrs(g_param)["cv_after_x3"] = Float32(cv_after[3])

            g_param2["on_fraction"]  = Float32.(on_frac[coarse_indices_in_full])
            g_param2["off_fraction"] = Float32.(off_frac[coarse_indices_in_full])
            attrs(g_param2)["accepted_nodes"] = accepted_nodes
            attrs(g_param2)["discarded_node"] = discarded_node

            final_on  = on_frac[end]
            final_off = off_frac[end]
            on_pv, on_pt, on_which, on_ratio     = gated_peak(on_frac,  times_full_vec, final_on)
            off_pv, off_pt, off_which, off_ratio = gated_peak(off_frac, times_full_vec, final_off)
            primary_ratio, primary_time, primary_state = primary_overshoot(
                on_ratio, on_pt, on_which, off_ratio, off_pt, off_which)

            push!(summary_df, (
                id, p_idx, accepted_nodes, discarded_node, Float32(threshold_equiv),
                Float32(ic_vals[1]), Float32(ic_vals[2]), Float32(ic_vals[3]),
                Float32(gmm_mean1_vec[1]), Float32(gmm_mean2_vec[1]),
                Float32(gmm_mean1_vec[2]), Float32(gmm_mean2_vec[2]),
                Float32(gmm_mean1_vec[3]), Float32(gmm_mean2_vec[3]),
                Float32(far_vec[1]), Float32(far_vec[2]), Float32(far_vec[3]),
                Float32(primary_ratio), Float32(primary_time), primary_state,
                Float32(stability_time),
                Float32(cv_before[1]), Float32(cv_before[2]), Float32(cv_before[3]),
                Float32(cv_after[1]), Float32(cv_after[2]), Float32(cv_after[3]),
                Float32(mean_path_vec[1]), Float32(std_path_vec[1]), Float32(median_path_vec[1]),
                Float32(mean_path_vec[2]), Float32(std_path_vec[2]), Float32(median_path_vec[2]),
                Float32(mean_path_vec[3]), Float32(std_path_vec[3]), Float32(median_path_vec[3])
            ))

            push!(overshoot_df, (
                id, p_idx, accepted_nodes,
                Float32(final_on), Float32(on_pv), Float32(on_pt), on_which, Float32(on_ratio),
                Float32(final_off), Float32(off_pv), Float32(off_pt), off_which, Float32(off_ratio)
            ))

            logmsg("net_$id param_$p_idx → MULTIMODAL | nodes=$accepted_nodes stability_time=$stability_time")
          catch err
              logmsg("net_$id param_$p_idx → ERROR: $err → skipped")
          end
        end
        logmsg("-> net_$id done")

        GC.gc()
    end
end
end
CSV.write(csv_file, summary_df)
CSV.write(overshoot_csv_file, overshoot_df)
total_time = Dates.canonicalize(now() - start_time)
logmsg("FINISHED! Found $(nrow(summary_df)) multimodal parameter sets.")
logmsg("Overshoot rows: $(nrow(overshoot_df))")
logmsg("Results saved to: $csv_file")
logmsg("Overshoot results saved to: $overshoot_csv_file")
logmsg("Total runtime: $total_time")
