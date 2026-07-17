import Pkg

JULIA_ENV_DIR = joinpath(@__DIR__, "julia_files")
Pkg.activate(JULIA_ENV_DIR)

using Dates
using JSON
using LinearAlgebra
using Statistics
using NLsolve
using JuMP
using COSMO
import MathOptInterface as MOI

RESULTS_BASE_DIR = joinpath(@__DIR__, "optimization_results")
RESULTS_SUBFOLDER = "periodic_chain_N12"
RESULTS_DIR = joinpath(RESULTS_BASE_DIR, RESULTS_SUBFOLDER)

function json_ready_matrix(A::AbstractMatrix)
    return [collect(row) for row in eachrow(A)]
end

function save_optimization_results(
    result,
    K::AbstractMatrix,
    omega::AbstractVector;
    theta0::Union{Nothing,AbstractVector} = nothing,
    output_dir::AbstractString = RESULTS_DIR,
    run_name::AbstractString = "kuramoto_noise",
    metadata::AbstractDict = Dict{String,Any}(),
)
    mkpath(output_dir)

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    payload = Dict{String,Any}(
        "metadata" => merge(
            Dict{String,Any}(
                "run_name" => run_name,
                "timestamp" => timestamp,
                "julia_version" => string(VERSION),
            ),
            metadata,
        ),
        "initial_conditions" => Dict{String,Any}(
            "N" => length(omega),
            "K" => json_ready_matrix(K),
            "omega" => collect(omega),
            "theta0" => theta0 === nothing ? nothing : collect(theta0),
        ),
        "fixed_point" => Dict{String,Any}(
            "theta_bar" => collect(result.theta_bar),
            "omega_centered" => collect(result.omega_centered),
            "residual" => collect(result.fixed_point_residual),
            "max_residual_inf_norm" => norm(result.fixed_point_residual, Inf),
        ),
        "optimization" => Dict{String,Any}(
            "C_opt" => json_ready_matrix(result.C_opt),
            "E_opt" => json_ready_matrix(result.E_opt),
            "objective" => result.objective,
            "cosmo_status" => string(result.cosmo_status),
            "lyapunov_residual" => json_ready_matrix(result.lyapunov_residual),
            "max_lyapunov_residual_inf_norm" => norm(result.lyapunov_residual, Inf),
            "C_eigenvalues" => collect(result.C_eigenvalues),
            "L_eigenvalues" => collect(result.L_eigenvalues),
        ),
        "linearized_system" => Dict{String,Any}(
            "L" => json_ready_matrix(result.L),
            "H" => json_ready_matrix(result.H),
        ),
    )

    run_path = joinpath(output_dir, "$(run_name)_$(timestamp).json")
    latest_path = joinpath(output_dir, "latest.json")
    open(run_path, "w") do io
        JSON.print(io, payload, 4)
        println(io)
    end
    open(latest_path, "w") do io
        JSON.print(io, payload, 4)
        println(io)
    end

    return (run_path = run_path, latest_path = latest_path)
end

"""
    solve_optimal_noise(K, omega; theta0=zeros(N-1), tol=1e-9, cosmo_tol=1e-7)

Given a symmetric coupling matrix `K` and natural frequencies `omega`, this function:

1. Centers `omega` so that `sum(omega) = 0`.
2. Finds a phase-locked Kuramoto fixed point `theta_bar` using NLsolve's Newton method.
3. Constructs the paper's linearized weighted Laplacian `L`.
4. Constructs the Hessian `H` of the squared Kuramoto order parameter.
5. Solves the SDP

       maximize    tr(H E)
       subject to  L E + E L = -C
                   C ⪰ 0
                   diag(C) = 1
                   C * 1 = 0
                   E * 1 = 0

   using JuMP + COSMO.

The constraints `diag(C)=1` and `C*1=0` reproduce the paper's fixed
uniform noise variance and centered-frame covariance.

Returns a named tuple containing `theta_bar`, `L`, `H`, `C_opt`, `E_opt`,
the objective value, and solver diagnostics.
"""
function solve_optimal_noise(
    K::AbstractMatrix{<:Real},
    omega::AbstractVector{<:Real};
    theta0::Union{Nothing,AbstractVector{<:Real}} = nothing,
    tol::Real = 1e-9,
    cosmo_tol::Real = 1e-7,
)
    N = length(omega)

    size(K) == (N, N) || throw(DimensionMismatch("K must be an N×N matrix, where N = length(omega)."))
    N >= 2 || throw(ArgumentError("At least two oscillators are required."))
    isapprox(K, K'; atol=tol, rtol=tol) || throw(ArgumentError("K must be symmetric for the paper's formulation."))

    Kf = Matrix{Float64}(K)
    wf = Vector{Float64}(omega)
    wf .-= mean(wf)

    # Gauge fixing: optimize N-1 coordinates and impose sum(theta_bar) = 0.
    function unpack_theta(x)
        theta = Vector{eltype(x)}(undef, N)
        theta[1:N-1] .= x
        theta[N] = -sum(x)
        return theta
    end

    function fixed_point_residual!(F, x)
        theta = unpack_theta(x)
        @inbounds for i in 1:N-1
            residual = wf[i]
            for j in 1:N
                residual += Kf[i, j] * sin(theta[j] - theta[i])
            end
            F[i] = residual
        end
        return nothing
    end

    x0 = theta0 === nothing ? zeros(N - 1) : Vector{Float64}(theta0)
    length(x0) == N - 1 || throw(DimensionMismatch("theta0 must have length N-1 because the mean phase is gauge-fixed."))

    root = nlsolve(
        fixed_point_residual!,
        x0;
        method = :newton,
        autodiff = :forward,
        ftol = tol,
        xtol = tol,
        iterations = 1_000,
    )

    NLsolve.converged(root) || error(
        "Newton solve did not converge. Try a better theta0, stronger coupling, or a continuation in K. " *
        "Residual norm = $(norm(root.fvec))."
    )

    theta_bar = unpack_theta(root.zero)

    # Verify the full N-equation fixed-point system.
    full_residual = similar(theta_bar)
    @inbounds for i in 1:N
        full_residual[i] = wf[i] + sum(Kf[i, j] * sin(theta_bar[j] - theta_bar[i]) for j in 1:N)
    end
    norm(full_residual, Inf) <= 100tol || error(
        "The reduced Newton system converged, but the full fixed-point residual is too large: " *
        "$(norm(full_residual, Inf)). Check that K is symmetric and omega is balanced."
    )

    # Paper's linearized dynamics:
    # L_ij = K_ij cos(theta_j - theta_i), i != j
    # L_ii = -sum_n K_in cos(theta_n - theta_i)
    L = zeros(Float64, N, N)
    @inbounds for i in 1:N
        rowsum = 0.0
        for j in 1:N
            if i != j
                wij = Kf[i, j] * cos(theta_bar[j] - theta_bar[i])
                L[i, j] = wij
                rowsum += wij
            end
        end
        L[i, i] = -rowsum
    end
    L = Matrix(Symmetric((L + L') / 2))

    # Hessian of R^2 at theta_bar:
    # H_ij = (2/N^2) [cos(theta_i-theta_j)
    #          - delta_ij sum_n cos(theta_i-theta_n)]
    H = zeros(Float64, N, N)
    scale = 2.0 / N^2
    @inbounds for i in 1:N
        for j in 1:N
            H[i, j] = scale * cos(theta_bar[i] - theta_bar[j])
        end
        H[i, i] -= scale * sum(cos(theta_bar[i] - theta_bar[n]) for n in 1:N)
    end
    H = Matrix(Symmetric((H + H') / 2))

    # Require stability on the centered subspace. L always has one zero mode.
    lambda = eigvals(Symmetric(L))
    nonzero_lambda = sort(lambda)[1:N-1]
    maximum(nonzero_lambda) < 1e-8 || @warn(
        "The fixed point may not be linearly stable on the centered subspace.",
        eigenvalues = lambda,
    )

    model = Model(
        optimizer_with_attributes(
            COSMO.Optimizer,
            "eps_abs" => Float64(cosmo_tol),
            "eps_rel" => Float64(cosmo_tol),
        ),
    )

    @variable(model, E[1:N, 1:N], Symmetric)
    @variable(model, C[1:N, 1:N], PSD)

    # Continuous Lyapunov equation: L*E + E*L = -C.
    @constraint(model, [i in 1:N, j in 1:N],
        sum(L[i, k] * E[k, j] + E[i, k] * L[k, j] for k in 1:N) + C[i, j] == 0
    )

    # The paper fixes uniform noise variances C_ii = 1.
    @constraint(model, [i in 1:N], C[i, i] == 1.0)

    # Centered-frame constraints.
    @constraint(model, [i in 1:N], sum(C[i, j] for j in 1:N) == 0.0)
    @constraint(model, [i in 1:N], sum(E[i, j] for j in 1:N) == 0.0)

    # Since H and E are symmetric, this equals tr(H*E).
    @objective(model, Max, sum(H[i, j] * E[i, j] for i in 1:N, j in 1:N))

    optimize!(model)

    status = termination_status(model)
    status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) || error("COSMO failed with termination status $status")

    E_opt = Matrix(value.(E))
    C_opt = Matrix(value.(C))

    return (
        theta_bar = theta_bar,
        omega_centered = wf,
        fixed_point_residual = full_residual,
        L = L,
        H = H,
        C_opt = C_opt,
        E_opt = E_opt,
        objective = objective_value(model),
        newton_result = root,
        cosmo_status = status,
        lyapunov_residual = L * E_opt + E_opt * L + C_opt,
        C_eigenvalues = eigvals(Symmetric(C_opt)),
        L_eigenvalues = lambda,
    )
end


# -----------------------------------------------------------------------------
# Example: a periodic nearest-neighbor chain
# -----------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    N = 12
    coupling = 0.5

    K = zeros(N, N)
    for i in 1:N
        jnext = mod1(i + 1, N)
        K[i, jnext] = coupling
        K[jnext, i] = coupling
    end

    # Example balanced natural frequencies.
    omega = collect(range(-0.04, 0.04; length=N))
    omega .-= mean(omega)

    result = solve_optimal_noise(K, omega)
    saved_files = save_optimization_results(
        result,
        K,
        omega;
        output_dir = RESULTS_DIR,
        run_name = "periodic_chain_N$(N)",
        metadata = Dict{String,Any}(
            "network" => "periodic nearest-neighbor chain",
            "coupling" => coupling,
        ),
    )

    println("theta_bar = ")
    display(result.theta_bar)

    println("\nOptimal covariance C = ")
    display(result.C_opt)

    println("\nObjective tr(H E) = ", result.objective)
    println("Maximum fixed-point residual = ", norm(result.fixed_point_residual, Inf))
    println("Maximum Lyapunov residual = ", norm(result.lyapunov_residual, Inf))
    println("Minimum eigenvalue of C = ", minimum(result.C_eigenvalues))
    println("Saved timestamped results to ", saved_files.run_path)
    println("Saved latest results to ", saved_files.latest_path)
end
