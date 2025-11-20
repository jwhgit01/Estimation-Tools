using LinearAlgebra
import DifferentialEquations as ODE
import StochasticDiffEq as SDE
using JLD2
import MAT
include("./models/rigid_body.jl")
using BenchmarkTools

# Dimensions
const nx = 6
const nw = 3
const nz = 6

# Scale diffusion matrix by Sigma, the square root of the process noise PSD
Sigma = Diagonal([1.0, 1.0, 0.5])

# Measurement noise covariance
R = 0.01*Matrix{Float64}(I,nz,nz)

# Rigid body moment of inertia
Ibody = [1.0 0 0.1;0 1.5 0;0.1 0 0.5]
params = (I = Ibody,)

# Final time
T = 10

# Sample times
dt = 0.01
t = 0:dt:T
N = length(t)

# Euler-Maruyama time step
dt_EM = 1.0e-4

# Drift and diffusion functions from the RigidBody module
function drift!(dx, x, params, t)
    f,_ = RigidBody.driftModel(t, x, nothing, params)
    dx .= f
end
function diffusion!(dG, x, params, t)
    D,_ = RigidBody.diffusionModel(t, x, nothing, params)
    dG .= D*Sigma
end

# Initial condition
x0 = [zeros(3); randn(3)]

# Set up the SDE problem
prob = SDE.SDEProblem(drift!, diffusion!, x0, (0,T), params, noise_rate_prototype = zeros(nx,nw), save_noise = true)

# Solve using the Euler-Maruyama scheme
sol = SDE.solve(prob, SDE.EM(), dt = dt_EM)

# Get states at discrete sample times, t.
x = sol(t).u

# Measurement noise
L = cholesky(R).L
v = [L * randn(nz) for _ in 1:N]  # Vector{Vector{Float64}}, each length nz

# Measurement model
y = [Vector{Float64}(undef, nz) for _ in 1:N]
for k in 1:N
    yk, _ = RigidBody.measurementModel(t[k],x[k],nothing,params)
    y[k] = yk
end

# Add measurement noise
z = y + v

# Save data to a JLD2 file
outfile = joinpath(@__DIR__,"data","JuliaData_RigidBody.jld2")
@save outfile Sigma R params x0 sol t x y z

# Also save compatible data to MAT file
vars = Dict(
    "Sigma" => Sigma,
    "R" => R,
    "I" => Ibody,
    "x0" => x0,
    "t" => hcat(t),
    "x" => hcat(x...),
    "y" => hcat(y...),
    "z" => hcat(z...)
)
outfile = joinpath(@__DIR__,"data","JuliaData_RigidBody.mat")
MAT.matwrite(outfile, vars)
