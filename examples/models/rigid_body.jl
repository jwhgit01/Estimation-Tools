module RigidBody

using LinearAlgebra

export driftModel, diffusionModel, measurementModel

function cpem(a)
    return [0 -a[3] a[2]; a[3] 0 -a[1]; -a[2] a[1] 0]
end

function driftModel(t, x, u, params)
    I = params.I
    Theta = x[1:3]
    omega = x[4:6]

    phi = Theta[1]
    theta = Theta[2]

    L_IB = [1 sin(phi)*tan(theta) cos(phi)*tan(theta); 0 cos(phi) -sin(phi); 0 sin(phi)/cos(theta) cos(phi)/cos(theta)]

    f_theta = L_IB * omega
    f_omega = I \ cross(I * omega, omega)
    f = vcat(f_theta, f_omega)

    A = zeros(eltype(f), 6, 6)
    q = omega[2]
    r = omega[3]

    A[1, 1] = (sin(theta) * (q * cos(phi) - r * sin(phi))) / cos(theta)
    A[1, 2] = (r * cos(phi) + q * sin(phi)) / cos(theta)^2
    A[1:3, 4:6] .= L_IB
    A[4:6, 4:6] .= I \ (cpem(I * omega) - cpem(omega) * I)

    return f, A
end

function diffusionModel(t, x, u, params)
    Theta = x[1:3]

    phi = Theta[1]
    theta = Theta[2]
    psi = Theta[3]

    R1 = [1.0 0.0 0.0;
          0.0 cos(phi) -sin(phi);
          0.0 sin(phi) cos(phi)]
    R2 = [cos(theta) 0.0 sin(theta);
          0.0 1.0 0.0;
          -sin(theta) 0.0 cos(theta)]
    R3 = [cos(psi) -sin(psi) 0.0;
          sin(psi) cos(psi) 0.0;
          0.0 0.0 1.0]

    R_IB = R3 * R2 * R1

    D = [zeros(3, 3); R_IB']

    T = eltype(x)
    J = zeros(T, 6, 3, 6)

    J[5:6, 1:3, 1] .= [sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)    cos(phi) * sin(psi) * sin(theta) - cos(psi) * sin(phi)    cos(phi) * cos(theta);
                       cos(phi) * sin(psi) - cos(psi) * sin(phi) * sin(theta)   -cos(phi) * cos(psi) - sin(phi) * sin(psi) * sin(theta)  -cos(theta) * sin(phi)]
    J[4:6, 1:3, 2] .= [        -cos(psi) * sin(theta)          -sin(psi) * sin(theta)           -cos(theta);
                        cos(psi) * cos(theta) * sin(phi)  cos(theta) * sin(phi) * sin(psi)  -sin(phi) * sin(theta);
                        cos(phi) * cos(psi) * cos(theta)  cos(phi) * cos(theta) * sin(psi)  -cos(phi) * sin(theta)]
    J[4:6, 1:2, 3] .= [                 -cos(theta) * sin(psi)                   cos(psi) * cos(theta);
                      -cos(phi) * cos(psi) - sin(phi) * sin(psi) * sin(theta)   cos(psi) * sin(phi) * sin(theta) - cos(phi) * sin(psi);
                       cos(psi) * sin(phi) - cos(phi) * sin(psi) * sin(theta)   sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)]

    return D, J
end

function measurementModel(t, x, u, params)
    Theta = x[1:3]

    phi = Theta[1]
    theta = Theta[2]
    psi = Theta[3]

    R1 = [1.0 0.0 0.0;
          0.0 cos(phi) -sin(phi);
          0.0 sin(phi) cos(phi)]
    R2 = [cos(theta) 0.0 sin(theta);
          0.0 1.0 0.0;
          -sin(theta) 0.0 cos(theta)]
    R3 = [cos(psi) -sin(psi) 0.0;
          sin(psi) cos(psi) 0.0;
          0.0 0.0 1.0]

    R_IB = R3 * R2 * R1

    y1 = R_IB' * [1.0, 0.0, 0.0]
    y2 = R_IB' * [0.0, 0.0, 1.0]
    y = vcat(y1, y2)

    T = eltype(x)
    H = zeros(T, 6, 6)
    H[1, 2] = -cos(psi) * sin(theta)
    H[1, 3] = -cos(theta) * sin(psi)
    H[2, 1] = sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)
    H[2, 2] = cos(psi) * cos(theta) * sin(phi)
    H[2, 3] = -cos(phi) * cos(psi) - sin(phi) * sin(psi) * sin(theta)
    H[3, 1] = cos(phi) * sin(psi) - cos(psi) * sin(phi) * sin(theta)
    H[3, 2] = cos(phi) * cos(psi) * cos(theta)
    H[3, 3] = cos(psi) * sin(phi) - cos(phi) * sin(psi) * sin(theta)
    H[4, 2] = -cos(theta)
    H[5, 1] = cos(phi) * cos(theta)
    H[5, 2] = -sin(phi) * sin(theta)
    H[6, 1] = -cos(theta) * sin(phi)
    H[6, 2] = -cos(phi) * sin(theta)

    return y, H
end

end # module
