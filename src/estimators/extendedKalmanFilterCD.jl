module ExtendedKalmanFilterCD

using LinearAlgebra
using LinearAlgebra: axpy!, rmul!, ldiv!

export ExtendedKalmanFilterCD, simulate, smooth, predict, correct, rts, Q, R

abstract type StateEstimatorCD end

mutable struct ExtendedKalmanFilterCD{F,D,H,Q,R} <: StateEstimatorCD
    drift::F
    diffusion::D
    measurement_model::H
    process_noise_psd::Q
    measurement_noise_covariance::R
    n_rk::Int
    display_period::Float64
    function ExtendedKalmanFilterCD(drift::F, diffusion::D, measurement_model::H,
                                    process_noise_psd::Q, measurement_noise_covariance::R;
                                    n_rk::Integer=10, display_period::Real=0.0) where {F,D,H,Q,R}
        new{F,D,H,Q,R}(drift, diffusion, measurement_model,
                       process_noise_psd, measurement_noise_covariance,
                       Int(n_rk), float(display_period))
    end
end

ExtendedKalmanFilterCD(f, D, h, Q, R) = ExtendedKalmanFilterCD(f, D, h, Q, R; n_rk=10, display_period=0.0)

function Q(filter::ExtendedKalmanFilterCD, t)
    psd = filter.process_noise_psd
    return psd isa Function ? psd(t) : psd
end

function R(filter::ExtendedKalmanFilterCD, k)
    meas = filter.measurement_noise_covariance
    if meas isa Function
        return meas(k)
    elseif ndims(meas) == 3 && size(meas, 3) > 1
        return @view meas[:, :, k + 1]
    else
        return meas
    end
end

function disp_iter(filter::ExtendedKalmanFilterCD, t)
    period = filter.display_period
    if period > 0
        remainder = mod(t, period)
        if isapprox(remainder, 0; atol=10 * eps(Float64), rtol=0)
            println("t = $(round(t; digits=2))")
        end
    end
    return nothing
end

mutable struct PredictWorkspace{TX,TP,TD}
    xstage::TX
    xbase::TX
    k1::TX
    k2::TX
    k3::TX
    k4::TX
    Pstage::TP
    Pbase::TP
    Pk1::TP
    Pk2::TP
    Pk3::TP
    Pk4::TP
    Pdot::TP
    tmpDQ::TD
end

function PredictWorkspace(x::AbstractVector, P::AbstractMatrix, D::AbstractMatrix)
    ny = size(D, 2)
    return PredictWorkspace(similar(x), similar(x), similar(x), similar(x), similar(x),
                             similar(x), similar(P), similar(P), similar(P), similar(P),
                             similar(P), similar(P), similar(P), Matrix{eltype(P)}(undef, size(P, 1), ny))
end

function compute_Pdot!(Pdot, tmpDQ, A, P, D, Qcur)
    mul!(Pdot, A, P)
    mul!(Pdot, P, A'; α=1.0, β=1.0)
    if Qcur isa UniformScaling
        copyto!(tmpDQ, D)
        rmul!(tmpDQ, Qcur.λ)
    else
        mul!(tmpDQ, D, Qcur)
    end
    mul!(Pdot, tmpDQ, D'; α=1.0, β=1.0)
    return Pdot
end

mutable struct CorrectWorkspace{TmatNZ,TmatNX,TmatNXN,TvecNZ,TvecNX}
    HP::TmatNZ
    solveHP::TmatNZ
    K::TmatNX
    KH::TmatNXN
    ImKH::TmatNXN
    tmpnxn::TmatNXN
    knu::TvecNX
    solve_vec::TvecNZ
    identity::TmatNXN
end

function CorrectWorkspace(nx::Integer, nz::Integer, Tstate, Tmeas)
    HP = Matrix{Tstate}(undef, nz, nx)
    solveHP = similar(HP)
    K = Matrix{Tstate}(undef, nx, nz)
    KH = Matrix{Tstate}(undef, nx, nx)
    ImKH = Matrix{Tstate}(undef, nx, nx)
    tmpnxn = Matrix{Tstate}(undef, nx, nx)
    knu = Vector{Tstate}(undef, nx)
    solve_vec = Vector{Tmeas}(undef, nz)
    identity = Matrix{Tstate}(I, nx, nx)
    return CorrectWorkspace(HP, solveHP, K, KH, ImKH, tmpnxn, knu, solve_vec, identity)
end

function add_measurement_covariance!(Sk, Rkp1)
    if Rkp1 isa UniformScaling
        λ = Rkp1.λ
        @inbounds for i in 1:size(Sk, 1)
            Sk[i, i] += λ
        end
    else
        Sk .+= Rkp1
    end
    return Sk
end

function symmetrize!(A::AbstractMatrix)
    @inbounds for j in 1:size(A, 2)
        for i in (j + 1):size(A, 1)
            v = (A[i, j] + A[j, i]) * 0.5
            A[i, j] = v
            A[j, i] = v
        end
    end
    return A
end


function simulate(filter::ExtendedKalmanFilterCD, t, z, u, xhat0, P0, params)
    N = size(z, 1)
    nx = length(xhat0)
    nz = size(z, 2)

    if length(t) != N
        throw(ArgumentError("Time history must contain $N samples to match the measurements."))
    end

    xhat = zeros(eltype(xhat0), N, nx)
    P = zeros(eltype(P0), nx, nx, N)
    nu = zeros(eltype(z), N, nz)
    epsnu = zeros(promote_type(Float64, real(eltype(z))), N)

    xhat[1, :] .= xhat0
    P[:, :, 1] .= P0

    Tcov = promote_type(Float64, real(eltype(P0)))
    maxsigdig = -floor(Int, log10(eps(Tcov)))
    sigdig = maxsigdig

    if u === nothing || isempty(u)
        u = zeros(eltype(xhat0), N, 0)
    elseif ndims(u) == 1
        u = reshape(u, N, :)
    end

    if size(u, 1) != N
        throw(ArgumentError("Input history must have $N rows to match the time vector."))
    end

    @views u0 = u[1, :]
    D0, _ = filter.diffusion(t[1], xhat0, u0, params)
    predict_ws = PredictWorkspace(xhat0, P0, D0)
    correct_ws = CorrectWorkspace(nx, nz, promote_type(eltype(P0), eltype(xhat0)), promote_type(eltype(z), Float64))

    xk = similar(xhat0)
    xbark = similar(xhat0)
    xnext = similar(xhat0)
    zk = Vector{eltype(z)}(undef, nz)
    nuk_storage = similar(zk)
    Sk = Matrix{promote_type(eltype(P0), eltype(z))}(undef, nz, nz)
    Pk = similar(P0)
    Pbark = similar(P0)
    Pnext = similar(P0)
    uk = Vector{eltype(u)}(undef, size(u, 2))

    cholS = nothing

    for k in 0:(N - 2)
        disp_iter(filter, t[k + 2])
        ik = k + 1

        tk = t[ik]
        tkp1 = t[ik + 1]

        @views copyto!(xk, xhat[ik, :])
        @views copyto!(Pk, P[:, :, ik])
        @views copyto!(uk, u[ik, :])
        predict!(xbark, Pbark, filter, tk, tkp1, xk, uk, Pk, params, predict_ws)

        @views copyto!(uk, u[ik + 1, :])
        @views copyto!(zk, z[ik + 1, :])
        cholS = correct!(xnext, Pnext, nuk_storage, Sk, filter, k + 1, zk, xbark, uk, Pbark, params, correct_ws)

        @views copyto!(xhat[ik + 1, :], xnext)
        @views copyto!(P[:, :, ik + 1], Pnext)
        @views copyto!(nu[ik + 1, :], nuk_storage)
        epsnu[ik + 1] = real(dot(nuk_storage, correct_ws.solve_vec))

        condS = cond(cholS)
        if isfinite(condS)
            sigdigkp1 = maxsigdig - floor(Int, log10(condS))
            sigdig = min(sigdig, sigdigkp1)
        else
            sigdig = 0
        end
    end

    return xhat, P, nu, epsnu, sigdig
end

function simulate(filter::ExtendedKalmanFilterCD, t, z, xhat0, P0, params)
    simulate(filter, t, z, nothing, xhat0, P0, params)
end

function smooth(filter::ExtendedKalmanFilterCD, t, z, u, xhat0, P0, params)
    N = size(z, 1)
    if u === nothing || isempty(u)
        u = zeros(eltype(xhat0), N, 0)
    elseif ndims(u) == 1
        u = reshape(u, N, :)
    end

    if size(u, 1) != N
        throw(ArgumentError("Input history must have $N rows to match the time vector."))
    end

    xhat, P, _, _, _ = simulate(filter, t, z, u, xhat0, P0, params)
    xs = copy(xhat)
    Ps = copy(P)

    Psk = copy(view(P, :, :, N))
    xsk = copy(view(xhat, N, :))

    if N > 1
        for k in reverse(1:(N - 1))
            disp_iter(filter, t[k + 1])
            tk = t[k + 1]
            tkm1 = t[k]
            xhatk = copy(view(xhat, k + 1, :))
            Pk = copy(view(P, :, :, k + 1))
            uk = copy(view(u, k + 1, :))
            xskm1, Pskm1 = rts(filter, tk, tkm1, xsk, uk, Psk, xhatk, Pk, params)

            xs[k, :] .= xskm1
            Ps[:, :, k] .= Pskm1

            xsk = xskm1
            Psk = Pskm1
        end
    end

    return xs, Ps, xhat, P
end

function smooth(filter::ExtendedKalmanFilterCD, t, z, xhat0, P0, params)
    smooth(filter, t, z, nothing, xhat0, P0, params)
end

function predict!(xbarkp1, Pbarkp1, filter::ExtendedKalmanFilterCD, tk, tkp1, xhatk, uk, Pk, params, workspace::PredictWorkspace)
    copyto!(xbarkp1, xhatk)
    copyto!(Pbarkp1, Pk)
    tcur = tk
    delt = (tkp1 - tk) / filter.n_rk

    for _ in 1:filter.n_rk
        copyto!(workspace.xbase, xbarkp1)
        copyto!(workspace.Pbase, Pbarkp1)

        f1, A1 = filter.drift(tcur, workspace.xbase, uk, params)
        D1, _ = filter.diffusion(tcur, workspace.xbase, uk, params)
        copyto!(workspace.k1, f1)
        rmul!(workspace.k1, delt)
        compute_Pdot!(workspace.Pdot, workspace.tmpDQ, A1, workspace.Pbase, D1, Q(filter, tcur))
        copyto!(workspace.Pk1, workspace.Pdot)
        rmul!(workspace.Pk1, delt)

        copyto!(workspace.xstage, workspace.xbase)
        axpy!(0.5, workspace.k1, workspace.xstage)
        copyto!(workspace.Pstage, workspace.Pbase)
        axpy!(0.5, workspace.Pk1, workspace.Pstage)

        f2, A2 = filter.drift(tcur + 0.5 * delt, workspace.xstage, uk, params)
        D2, _ = filter.diffusion(tcur + 0.5 * delt, workspace.xstage, uk, params)
        copyto!(workspace.k2, f2)
        rmul!(workspace.k2, delt)
        compute_Pdot!(workspace.Pdot, workspace.tmpDQ, A2, workspace.Pstage, D2, Q(filter, tcur + 0.5 * delt))
        copyto!(workspace.Pk2, workspace.Pdot)
        rmul!(workspace.Pk2, delt)

        copyto!(workspace.xstage, workspace.xbase)
        axpy!(0.5, workspace.k2, workspace.xstage)
        copyto!(workspace.Pstage, workspace.Pbase)
        axpy!(0.5, workspace.Pk2, workspace.Pstage)

        f3, A3 = filter.drift(tcur + 0.5 * delt, workspace.xstage, uk, params)
        D3, _ = filter.diffusion(tcur + 0.5 * delt, workspace.xstage, uk, params)
        copyto!(workspace.k3, f3)
        rmul!(workspace.k3, delt)
        compute_Pdot!(workspace.Pdot, workspace.tmpDQ, A3, workspace.Pstage, D3, Q(filter, tcur + 0.5 * delt))
        copyto!(workspace.Pk3, workspace.Pdot)
        rmul!(workspace.Pk3, delt)

        copyto!(workspace.xstage, workspace.xbase)
        axpy!(1.0, workspace.k3, workspace.xstage)
        copyto!(workspace.Pstage, workspace.Pbase)
        axpy!(1.0, workspace.Pk3, workspace.Pstage)

        f4, A4 = filter.drift(tcur + delt, workspace.xstage, uk, params)
        D4, _ = filter.diffusion(tcur + delt, workspace.xstage, uk, params)
        copyto!(workspace.k4, f4)
        rmul!(workspace.k4, delt)
        compute_Pdot!(workspace.Pdot, workspace.tmpDQ, A4, workspace.Pstage, D4, Q(filter, tcur + delt))
        copyto!(workspace.Pk4, workspace.Pdot)
        rmul!(workspace.Pk4, delt)

        copyto!(xbarkp1, workspace.xbase)
        axpy!(1 / 6, workspace.k1, xbarkp1)
        axpy!(1 / 3, workspace.k2, xbarkp1)
        axpy!(1 / 3, workspace.k3, xbarkp1)
        axpy!(1 / 6, workspace.k4, xbarkp1)

        copyto!(Pbarkp1, workspace.Pbase)
        axpy!(1 / 6, workspace.Pk1, Pbarkp1)
        axpy!(1 / 3, workspace.Pk2, Pbarkp1)
        axpy!(1 / 3, workspace.Pk3, Pbarkp1)
        axpy!(1 / 6, workspace.Pk4, Pbarkp1)

        tcur += delt
    end

    return xbarkp1, Pbarkp1
end

function predict(filter::ExtendedKalmanFilterCD, tk, tkp1, xhatk, uk, Pk, params)
    D0, _ = filter.diffusion(tk, xhatk, uk, params)
    workspace = PredictWorkspace(xhatk, Pk, D0)
    xbark = similar(xhatk)
    Pbark = similar(Pk)
    predict!(xbark, Pbark, filter, tk, tkp1, xhatk, uk, Pk, params, workspace)
    return xbark, Pbark
end

function correct!(xhatkp1, Pkp1, nukp1, Skp1, filter::ExtendedKalmanFilterCD, kp1, zkp1, xbarkp1, ukp1, Pbarkp1, params, workspace::CorrectWorkspace)
    zbarkp1, Hkp1 = filter.measurement_model(kp1, xbarkp1, ukp1, params)
    copyto!(nukp1, zkp1)
    axpy!(-1.0, zbarkp1, nukp1)

    mul!(workspace.HP, Hkp1, Pbarkp1)
    mul!(Skp1, workspace.HP, Hkp1'; α=1.0, β=0.0)
    Rkp1 = R(filter, kp1)
    add_measurement_covariance!(Skp1, Rkp1)

    copyto!(workspace.solveHP, workspace.HP)
    Skchol = cholesky(Symmetric(Skp1))
    ldiv!(Skchol, workspace.solveHP)
    permutedims!(workspace.K, workspace.solveHP)

    copyto!(workspace.solve_vec, nukp1)
    ldiv!(Skchol, workspace.solve_vec)

    mul!(workspace.knu, workspace.K, nukp1)
    copyto!(xhatkp1, xbarkp1)
    axpy!(1.0, workspace.knu, xhatkp1)

    mul!(workspace.KH, workspace.K, Hkp1)
    copyto!(workspace.ImKH, workspace.identity)
    axpy!(-1.0, workspace.KH, workspace.ImKH)

    mul!(Pkp1, workspace.ImKH, Pbarkp1)
    mul!(Pkp1, Pkp1, workspace.ImKH'; α=1.0, β=0.0)

    if Rkp1 isa UniformScaling
        mul!(workspace.tmpnxn, workspace.K, workspace.K'; Rkp1.λ, 0.0)
    else
        mul!(workspace.tmpnxn, workspace.K, Rkp1)
        mul!(workspace.tmpnxn, workspace.tmpnxn, workspace.K'; α=1.0, β=0.0)
    end
    Pkp1 .+= workspace.tmpnxn
    symmetrize!(Pkp1)

    return Skchol
end

function correct(filter::ExtendedKalmanFilterCD, kp1, zkp1, xbarkp1, ukp1, Pbarkp1, params)
    nx = length(xbarkp1)
    nz = length(zkp1)
    workspace = CorrectWorkspace(nx, nz, promote_type(eltype(Pbarkp1), eltype(xbarkp1)), promote_type(eltype(zkp1), Float64))
    xhat = similar(xbarkp1)
    P = similar(Pbarkp1)
    nu = similar(zkp1)
    Sk = Matrix{promote_type(eltype(Pbarkp1), eltype(zkp1))}(undef, nz, nz)
    correct!(xhat, P, nu, Sk, filter, kp1, zkp1, xbarkp1, ukp1, Pbarkp1, params, workspace)
    return xhat, P, nu, Sk
end

function rts(filter::ExtendedKalmanFilterCD, tk, tkm1, xsk, uk, Psk, xhatk, Pk, params)
    xs = copy(xsk)
    Ps = copy(Psk)
    tcur = tk
    dt = (tkm1 - tk) / filter.n_rk
    Pk_factor = cholesky(Symmetric(Pk))

    for _ in 1:filter.n_rk
        f, A = filter.drift(tcur, xs, uk, params)
        D = filter.diffusion(tcur, xs, uk, params)
        Qbar = D * Q(filter, tcur) * D'
        F = (Pk_factor \ Qbar')'
        Psdot = (A + F) * Ps + Ps * (A + F)' - Qbar
        dxsa = (f + F * (xs - xhatk)) * dt
        dPsa = Psdot * dt

        f, A = filter.drift(tcur + 0.5 * dt, xs + 0.5 * dxsa, uk, params)
        D = filter.diffusion(tcur + 0.5 * dt, xs + 0.5 * dxsa, uk, params)
        Qbar = D * Q(filter, tcur + 0.5 * dt) * D'
        F = (Pk_factor \ Qbar')'
        Psdot = (A + F) * (Ps + 0.5 * dPsa) + (Ps + 0.5 * dPsa) * (A + F)' - Qbar
        dxsb = (f + F * (xs - xhatk)) * dt
        dPsb = Psdot * dt

        f, A = filter.drift(tcur + 0.5 * dt, xs + 0.5 * dxsb, uk, params)
        D = filter.diffusion(tcur + 0.5 * dt, xs + 0.5 * dxsb, uk, params)
        Qbar = D * Q(filter, tcur + 0.5 * dt) * D'
        F = (Pk_factor \ Qbar')'
        Psdot = (A + F) * (Ps + 0.5 * dPsb) + (Ps + 0.5 * dPsb) * (A + F)' - Qbar
        dxsc = (f + F * (xs - xhatk)) * dt
        dPsc = Psdot * dt

        f, A = filter.drift(tcur + dt, xs + dxsc, uk, params)
        D = filter.diffusion(tcur + dt, xs + dxsc, uk, params)
        Qbar = D * Q(filter, tcur + dt) * D'
        F = (Pk_factor \ Qbar')'
        Psdot = (A + F) * (Ps + dPsc) + (Ps + dPsc) * (A + F)' - Qbar
        dxsd = (f + F * (xs - xhatk)) * dt
        dPsd = Psdot * dt

        xs += (dxsa + 2 * (dxsb + dxsc) + dxsd) / 6
        Ps += (dPsa + 2 * (dPsb + dPsc) + dPsd) / 6
        tcur += dt
    end

    return xs, Ps
end

end # module
