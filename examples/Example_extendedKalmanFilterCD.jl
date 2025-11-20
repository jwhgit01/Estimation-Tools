using LinearAlgebra
using DifferentialEquations
using Interpolations
using JLD2
using Plots
include("./models/rigid_body.jl")
include("../src/estimators/extendedKalmanBucyFilter.jl")
using BenchmarkTools

# Load data
datafile = joinpath(@__DIR__,"data","JuliaData_RigidBody.jld2")
@load datafile Sigma R params x0 t x y z

# Dimensions
const nx = 6
const nw = 3
const nz = 6

# Process and measurement noise power spectral densities
Q = Sigma'*Sigma

# Measurement function
z_fun = LinearInterpolation(t,z)

# Build EKF problem
prob = extendedKalmanBucyFilter.EKBFProblem(
    RigidBody.driftModel,
    RigidBody.diffusionModel,
    RigidBody.measurementModel,
    z_fun,
    Q,
    R,
    params;
    n_x = nx
)

# Initial estimate and covariance
xhat0 = randn(6)
P0 = I(nx)

sol = extendedKalmanBucyFilter.filter(
    prob,
    xhat0,
    P0,
    (t[1], t[end]),
    Tsit5();
    reltol = 1e-4,
    abstol = 1e-8,
)

plot(sol.t,permutedims(hcat(sol.x̂...)))
plot!(t,permutedims(hcat(x...)); linestyle=:dot)
# plotfile = joinpath(@__DIR__,"Example_RigidBody_EKBF")
# Plots.pdf(plotfile)
