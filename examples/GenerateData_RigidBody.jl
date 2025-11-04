using LinearAlgebra
using Random
using MAT
using StochasticDiffEq
using DiffEqNoiseProcess

include(joinpath(@__DIR__, "models/rigid_body.jl"))

using .RigidBodyModel

function previous_hold(tsrc::AbstractVector, values::AbstractMatrix, tquery::AbstractVector)
    nsrc = length(tsrc)
    nstate = size(values, 2)
    result = zeros(eltype(values), length(tquery), nstate)
    idx = 1
    for (i, tq) in enumerate(tquery)
        while idx < nsrc && tsrc[idx + 1] <= tq + eps(eltype(tsrc))
            idx += 1
        end
        idx = min(idx, nsrc)
        result[i, :] .= values[idx, :]
    end
    return result
end

function stack_states(states)
    return permutedims(hcat(states...), (2, 1))
end

function rigid_body_drift!(dx, x, params, t)
    f, _ = driftModel_RigidBody(t, x, nothing, params)
    dx .= f
    return nothing
end

function rigid_body_diffusion!(dx, x, params, t)
    D, _ = diffusionModel_RigidBody(t, x, nothing, params)
    mul!(dx, D, params.Sigma)
    return nothing
end

function generate_rigid_body_data()
    Random.seed!(0)

    nx = 6
    nw = 3
    nz = 6

    Ibody = [1.0 0.0 0.1;
         0.0 1.5 0.0;
         0.1 0.0 0.5]
    params = (I = Ibody,)

    T = 10.0
    dt = 0.01
    t = collect(0.0:dt:T)
    N = length(t)
    dtEM = 1.0e-4
    tSim = collect(0.0:dtEM:T)
    NEM = length(tSim)

    W = zeros(Float64, NEM, nw)
    sqrt_dtEM = sqrt(dtEM)
    for k in 2:NEM
        W[k, :] .= W[k - 1, :]
        W[k, :] .+= sqrt_dtEM .* transpose(randn(nw))
    end

    Sigma = Diagonal([1.0, 1.0, 0.5])
    Sigma_matrix = Matrix(Sigma)

    params_aug = (I = Ibody, Sigma = Sigma_matrix)

    x0 = vcat(zeros(3), randn(3))

    ΔW = permutedims(diff(W; dims = 1), (2, 1))
    noise = NoiseGrid(tSim, ΔW; reset = false)

    prob = SDEProblem(rigid_body_drift!, rigid_body_diffusion!, copy(x0), (first(tSim), last(tSim)), params_aug; noise = noise)
    sol = solve(prob, EM(), dt = dtEM, saveat = tSim)

    xSim = stack_states(sol.u)
    x = previous_hold(tSim, xSim, t)

    wtil = zeros(Float64, NEM, nw)
    for k in 2:NEM
        Δwk = W[k, :] .- W[k - 1, :]
        wtil[k - 1, :] .= (Sigma_matrix * (Δwk ./ dtEM))'
    end

    R = 0.01 .* Matrix{Float64}(I, nz, nz)
    L = cholesky(R).L
    v = transpose(L * randn(nz, N))

    y = zeros(Float64, N, nz)
    for k in 1:N
        y[k, :] .= first(measurementModel_RigidBody(t[k], x[k, :], nothing, params))
    end
    z = y .+ v

    params_dt = merge(params, (dt = dt,))
    params_dict = Dict("I" => Ibody, "dt" => dt)

    w = previous_hold(tSim, W, t)

    ΔWdt = permutedims(diff(w; dims = 1), (2, 1))
    noise_dt = NoiseGrid(t, ΔWdt; reset = false)
    prob_dt = SDEProblem(rigid_body_drift!, rigid_body_diffusion!, copy(x0), (first(t), last(t)), merge(params_aug, (dt = dt,)); noise = noise_dt)
    sol_dt = solve(prob_dt, EM(), dt = dt, saveat = t)
    xd = stack_states(sol_dt.u)

    Q = Sigma_matrix * Sigma_matrix' * dt

    yd = zeros(Float64, N, nz)
    for k in 1:N
        yd[k, :] .= first(measurementModel_RigidBody(t[k], xd[k, :], nothing, params_dt))
    end
    zd = yd .+ v

    vars = Dict(
        "t" => t,
        "dt" => dt,
        "x" => x,
        "xd" => xd,
        "y" => y,
        "z" => z,
        "zd" => zd,
        "wtil" => wtil,
        "W" => W,
        "v" => v,
        "w" => w,
        "Sigma" => Matrix(Sigma),
        "Q" => Q,
        "R" => R,
        "params" => params_dict,
        "tSim" => tSim,
        "xSim" => xSim
    )

    outfile = joinpath(@__DIR__, "data", "ExampleData_RigidBody.mat")
    matwrite(outfile, vars)

    return outfile
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Saved data to " * generate_rigid_body_data())
end
