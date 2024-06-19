function [xs,P] = simpleParticleSmoother(t,z,u,f,hk,Q,R,xhat0,P0,Ns,nRK,params)
%simpleParticleSmoother 
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs basic particle smoothing for a given time
% history of measurments and the discrete-time nonlinear system,
%
%                   x(k+1) = f(k,x(k),u(k),v(k))                    (1)
%                     z(k) = h(k,x(k)) + w(k)                       (2)
%
% where v(k) is zero-mean, white noise with covariance Q(k) and w(k) is
% zero-mean, white noise with covariance R(k).
%
% Inputs:
%
%   t       The Nz x 1 sample time vector. If f is a discrete-time dynamic
%           model, t must be givenn as an empty array, [].
%
%   z       The N x nz time history of measurements.
%
%   u       The N x nu time history of system inputs (optional). If not
%           applicable set to an empty array, [].
% 
%   f       The function handle that computes either the continuous-time
%           dynamics if t is given as a vector of sample times or the
%           discrete-time dynamics if t is empty. The first line of f must
%           be in the form
%               [f,A,D] = nonlindyn(t,x,u,vtil,dervflag,params)
%           or
%               [fk,Fk,Gamk] = nonlindyn(k,xk,uk,vk,dervflag,params)
% 
%   h       The function handle that computes the modeled output of the
%           system. The first line of h must be in the form
%               [h,H] = measmodel(t,x,u,dervflag,params)
%   
%   Q,R     The discrete-time process and measurement noise covariance.
%           These may be specificed as constant matrices, ()x()xN
%           3-dimensional arrays, or function handles that are functions of
%           the sample number, k.
%
%   xhat0   The nx x 1 initial state estimate.
%
%   P0      The nx x nx symmetric positive definite initial state
%           estimation error covariance matrix.
%
%   Ns      The number of particles to be propogated though the dynamics
%           over the given time history.
%
%   nRK     The number of intermediate Runge-Kutta integrations steps used
%           in the discretization of the nonlinear dynamics. May be given
%           as an empty array to use the default number.
%
%   params  A struct of constants that get passed to the dynamics model and
%           measurement model functions.
%
%
%  
% Outputs:
%
%   xs      The N x nx array that contains the time history of the
%           smoothed state vector estimates.
%
%   P       The nx x nx x N array that contains the time history of the
%           estimation error covariance matrices.
%

% Check to see whether we have non-stationary noise, which may
% be prescribed by an array of matrices or a function handle that is a
% fucntion of the timestep/time.
if ~isa(R,'function_handle')
    if size(R,3) > 1, Rk = @(k) R(:,:,k+1); else, Rk = @(k) R; end
else
    Rk = R;
end
if ~isa(Q,'function_handle')
    if size(Q,3) > 1, Qk = @(k) Q(:,:,k+1); else, Qk = @(k) Q; end
else
    Qk = Q;
end

% number of runge-kutta integration steps
if isempty(nRK)
    nRK = 10;
end

% Get the problem dimensions and initialize the output arrays.
N = size(z,1);
nx = size(xhat0,1);
nv = size(Qk(1),1);
xs = zeros(N+1,nx);
P = zeros(nx,nx,N+1);
P(:,:,1) = P0;
logwtil = zeros(1,Ns);
Xbar = zeros(N,nx,Ns);
Xbar(1,:,:) = xhat0;

% Initialize intermediate variables that get populated for each particle.
Xkp1 = zeros(nx,Ns);

% if no inputs, set to zero
if isempty(u)
    u = zeros(N,1);
end

% Initialize the particles by sampling Ns samples independently from
% N(xhat(0), P(0)) to generate Xi(0) for i = 1,...,Ns. Also set wi0 = 1/Ns
% for all i = 1,...,Ns.
Sx0 = chol(P0)';
Xk = xhat0 + Sx0*randn(nx,Ns);

% This loop performs one model propagation step and one measurement
% update step per iteration.
for k = 1:N-1

    % Recall, arrays are 1-indexed, but the initial condition occurs at k=0
    kp1 = k+1;

    % Generate process noise values for the particles by sampling vi(k)
    % independently from N(0;Q(k)) for i = 1,...,Ns. This is done in
    % preparation for dynamic propagation of the particles.
    Sv = chol(Qk(k))';
    Vk = Sv*randn(nv,Ns);

    % Dynamically propagate the particles and store the log of their
    % un-normalized weights.
    uk = u(kp1,:).';
    ukp1 = u(kp1+1,:).';
    zkp1 = z(k+1,:).';
    Rkp1inv = inv(Rk(k+1));
    for ii = 1:Ns
        if isempty(t)
            Xkp1(:,ii) = feval(f,k,Xk(:,ii),uk,Vk(:,ii),0,params);
            zibarkp1 = feval(hk,k+1,Xkp1(:,ii),ukp1,0,params);
        else
            tk = t(k,1);
            tkp1 = t(k+1,1);
            Xkp1(:,ii) = c2dNonlinear(Xk(:,ii),uk,Vk(:,ii),tk,tkp1,nRK,f,0,params);
            zibarkp1 = feval(hk,tkp1,Xkp1(:,ii),ukp1,0,params);
        end

        % Compute the weight of each particle based on the time history of
        % the measurement error. The weight of each of these particles is
        % proportional to the exponential of negative one half the sum of
        % the weighted squares of the measurement errors, summed over the
        % entire trajectory.
        zitilkp1 = zkp1 - zibarkp1;
        logwtil(ii) = logwtil(ii) - 0.5*zitilkp1'*Rkp1inv*zitilkp1;

    end

    % Store particle trajectory and update Xk.
    Xbar(kp1+1,:,:) = Xkp1;
    Xk = Xkp1;

end

% Normalize the particle weights and compute the smoothed estimate.
wtil = exp(logwtil-max(logwtil));
w = wtil/sum(wtil);
for ii = 1:Ns
    xs = xs + w(ii)*(Xbar(:,:,ii));
end
for ii = 1:Ns
    for kp1 = 2:N+1
        xktil = (Xbar(kp1,:,ii)-xs(kp1,:)).';
        P(:,:,kp1) = P(:,:,kp1) + w(ii)*(xktil*xktil');
    end
end

end