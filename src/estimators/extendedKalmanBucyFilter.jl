module extendedKalmanBucyFilter

using LinearAlgebra
using DifferentialEquations

export EKFProblem, ekbf_ode!, filter

"""
    EKFProblem(
        drift_model,
        diffusion_model,
        measurement_model,
        meas_fun,
        Q,
        R,
        params;
        n_x,
        u_fun = t -> nothing,
    )

Container for a continuous-time EKF problem.

Arguments
---------
- `drift_model(t, x, u, params) -> (f, A)`:
    Your process model. Returns drift `f` and Jacobian `A = ∂f/∂x`.

- `diffusion_model(t, x, u, params) -> (D, J)`:
    Your diffusion model. Only `D` (the diffusion matrix) is used
    in the EKF Riccati equation. `J` is ignored here.

- `measurement_model(t, x, u, params) -> (yhat, H)`:
    Measurement model returning predicted measurement `yhat = h(t,x,u)`
    and Jacobian `H = ∂h/∂x`.

- `meas_fun(t) -> y_meas`:
    Function providing the actual measured `y_meas(t)` at time `t`
    (e.g., via interpolation of simulated/recorded data).

- `Q`: Process noise PSD matrix for `wtil(t)` in `dx = f + D*wtil`.
- `R`: Measurement noise covariance.
- `params`: Arbitrary parameter container passed through to your models.
- `n_x`: State dimension.
- `u_fun(t) -> u` (optional): Control input function. Defaults to `t -> nothing`.

The EKF will integrate `(x̂, P)` forward in time using DifferentialEquations.jl.
"""
struct EKBFProblem{F1,F2,F3,F4,F5,TP,TR,TRI,TC}
    drift_model::F1
    diffusion_model::F2
    measurement_model::F3
    meas_fun::F4
    u_fun::F5
    Q::TP
    R::TR
    Rinv::TRI
    params::TC
    nx::Int
end

function EKBFProblem(
    drift_model,
    diffusion_model,
    measurement_model,
    meas_fun,
    Q,
    R,
    params;
    n_x,
    u_fun = t -> nothing,
)
    Rinv = inv(R)
    return EKBFProblem(
        drift_model,
        diffusion_model,
        measurement_model,
        meas_fun,
        u_fun,
        Q,
        R,
        Rinv,
        params,
        n_x,
    )
end

"""
    ekbf_ode!(dζ, ζ, p::EKBFProblem, t)

In-place ODE for the augmented EKF state

    ζ = [x̂; vec(P)]

of length `nx + nx^2`. Used internally by `solve_continuous_ekf`, but you
can also build your own `ODEProblem` around it if you like.
"""
function ekbf_ode!(dζ, ζ, p::EKBFProblem, t)
    nx = p.nx

    # Unpack state estimate and covariance
    xhat = @view ζ[1:nx]
    P    = reshape(@view(ζ[nx+1:end]), nx, nx)

    dxhat    = @view dζ[1:nx]
    dP_flat  = @view dζ[nx+1:end]

    # Control input
    u = p.u_fun(t)

    # Process model
    f, A = p.drift_model(t, xhat, u, p.params)

    # Diffusion model (only D is needed here)
    D, _ = p.diffusion_model(t, xhat, u, p.params)

    # Measurement model evaluated at the estimate
    yhat, H = p.measurement_model(t, xhat, u, p.params)

    # Actual measurement at time t
    y_meas = p.meas_fun(t)
    ν = y_meas - yhat

    # Kalman gain K = P H' R^{-1}
    K = P * H' * p.Rinv

    # State estimate dynamics
    dxhat .= f + K * ν

    # Riccati equation:
    # dP/dt = A P + P A' + D Q D' - P H' R^{-1} H P
    # We use the algebraically equivalent form with K:
    # dP/dt = A P + P A' + D Q D' - K R K'
    dP = A * P + P * A' + D * p.Q * D' - K * p.R * K'

    dP_flat .= vec(dP)

    return nothing
end

"""
    solve_continuous_ekf(p::EKBFProblem, xhat0, P0, tspan, solver=Tsit5(); kwargs...)

Convenience function to build and solve an `ODEProblem` for the continuous-time EKF.

Returns a DifferentialEquations.jl solution object whose `.u` field contains
the stacked vector `[x̂; vec(P)]` at each time step.
"""
function filter(
    p::EKBFProblem,
    xhat0::AbstractVector,
    P0::AbstractMatrix,
    tspan,
    solver = Tsit5();
    kwargs...,
)
    nx = p.nx
    @assert length(xhat0) == nx "xhat0 length must equal nx"
    @assert size(P0,1) == nx && size(P0,2) == nx "P0 must be nx×nx"

    ζ0 = [xhat0; vec(P0)]

    prob = ODEProblem(ekbf_ode!, ζ0, tspan, p)
    sol  = solve(prob, solver; kwargs...)
    
    # Extract t, x̂(t), and P(t) from solution
    t = sol.t;
    N = length(t);
    x̂ = [ sol.u[k][1:nx] for k in 1:N ]
    P = [ reshape(sol.u[k][nx+1:end],nx,nx) for k in 1:N ]

    sol_ekbf = EKBFSolution(t, x̂, P)
    return sol_ekbf
end

struct EKBFSolution
    t::AbstractVector
    x̂::AbstractVector
    P::AbstractVector
end

end # module
