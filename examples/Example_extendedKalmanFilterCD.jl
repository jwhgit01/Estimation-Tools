using LinearAlgebra
using MAT

include(joinpath(@__DIR__, "../src/estimators/extendedKalmanFilterCD.jl"))
include(joinpath(@__DIR__, "models/rigid_body.jl"))

using .ExtendedKalmanFilterCD
using .RigidBodyModel

function fetch_field(obj, key::Symbol)
    if obj isa NamedTuple
        return getfield(obj, key)
    elseif Base.hasproperty(obj, key)
        return getproperty(obj, key)
    elseif obj isa AbstractDict
        if haskey(obj, key)
            return obj[key]
        end
        skey = string(key)
        if haskey(obj, skey)
            return obj[skey]
        end
    end
    for accessor in (Symbol, String)
        if hasmethod(getindex, Tuple{typeof(obj), accessor})
            skey = accessor === Symbol ? key : string(key)
            try
                return obj[skey]
            catch
            end
        end
    end
    error("Unable to find field $(key)")
end

function main()
    datafile = joinpath(@__DIR__, "data", "ExampleData_RigidBody.mat")
    data = matread(datafile)

    t = vec(data["t"])
    x = Array(data["x"])
    z = Array(data["z"])
    Sigma = Array(data["Sigma"])
    R = Array(data["R"])
    params_raw = data["params"]
    params = (; I = Matrix(fetch_field(params_raw, :I)))
    tSim = vec(data["tSim"])
    xSim = Array(data["xSim"])

    nx = size(x, 2)
    nz = size(z, 2)

    Q = Sigma * Sigma'
    ekf = ExtendedKalmanFilterCD.ExtendedKalmanFilterCD(driftModel_RigidBody,
                                                        diffusionModel_RigidBody,
                                                        measurementModel_RigidBody,
                                                        Q, R)

    xhat0 = zeros(eltype(x), nx)
    P0 = Matrix{eltype(x)}(I, nx, nx)

    xhat, P, nu, epsnu, sigdig = ExtendedKalmanFilterCD.simulate(ekf, t, z, nothing, xhat0, P0, params)

    epsx = zeros(eltype(x), length(t))
    for k in eachindex(t)
        exk = xhat[k, :] .- x[k, :]
        epsx[k] = dot(exk, P[:, :, k] \ exk)
    end

    println("Final significant digits estimate: $(sigdig)")
    println("Final state estimate: ", xhat[end, :])
    println("Final innovation norm: ", norm(nu[end, :]))

    return (; t = t, x = x, z = z, xhat = xhat, P = P, nu = nu,
            epsnu = epsnu, epsx = epsx, sigdig = sigdig,
            tSim = tSim, xSim = xSim)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
