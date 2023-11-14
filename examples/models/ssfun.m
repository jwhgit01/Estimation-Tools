function [sys,jac,bias] = ssfun(theta,ParameterIndex,Constants)

% Dimensions
nx = 2; % states
nu = 3; % inputs
nv = 2; % process noise
ny = 5; % outputs
nw = ny; % measurement noise
nth = size(theta,1); % parameters

% Known portion of the system 
sys.A = zeros(nx,nx);
sys.B = zeros(nx,nu);
sys.C = [zeros(ny-nx,nx); eye(nx)];
sys.D = zeros(ny,nu);
sys.F = zeros(nx,nv);
sys.G = diag([0.164516, 0.1487645, 0.2213919, 0.0824749, 0.4420580]);

% Jacobians of the system matrices
jac.A = zeros(nx,nx,nth);
jac.B = zeros(nx,nu,nth);
jac.C = zeros(ny,nx,nth);
jac.D = zeros(ny,nu,nth);
jac.F = zeros(nx,nv,nth);
jac.G = zeros(ny,nw,nth);

% State Equation
for ii = 1:nx
    for jj = 1:nx
        pp = ParameterIndex.A(ii,jj);
        if pp == 0
            continue
        end
        sys.A(ii,jj) = sys.A(ii,jj) + theta(pp,1);
        jac.A(ii,jj,pp) = true;
    end
    for jj = 1:nu
        pp = ParameterIndex.B(ii,jj);
        if pp == 0
            continue
        end
        sys.B(ii,jj) = sys.B(ii,jj) + theta(pp,1);
        jac.B(ii,jj,pp) = true;
    end
    for jj = 1:nv
        pp = ParameterIndex.F(ii,jj);
        if pp == 0
            continue
        end
        sys.F(ii,jj) = sys.F(ii,jj) + theta(pp,1);
        jac.F(ii,jj,pp) = true;
    end
end

% Measurement equation
for ii = 1:ny
    for jj = 1:nx
        pp = ParameterIndex.C(ii,jj);
        if pp == 0
            continue
        end
        sys.C(ii,jj) = sys.C(ii,jj) + theta(pp,1);
        jac.C(ii,jj,pp) = true;
    end
    for jj = 1:nu
        pp = ParameterIndex.D(ii,jj);
        if pp == 0
            continue
        end
        sys.D(ii,jj) = sys.D(ii,jj) + theta(pp,1);
        jac.D(ii,jj,pp) = true;
    end
    for jj = 1:nw
        pp = ParameterIndex.G(ii,jj);
        if pp == 0
            continue
        end
        sys.G(ii,jj) = sys.G(ii,jj) + theta(pp,1);
        jac.G(ii,jj,pp) = true;
    end
end

% Bias vectors
bias.x = theta(ParameterIndex.bx);
bias.y = theta(ParameterIndex.by);

return