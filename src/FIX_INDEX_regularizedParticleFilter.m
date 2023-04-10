function [xhat,P] = regularizedParticleFilter(t,z,u,f,hk,q,Q,R,xhat0,P0,Ns,NT,nRK,params)
%regularizedParticleFilter 
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs regularized particle filtering for a given time
% history of measurments and the discrete-time nonlinear system,
%
%                   x(k+1) = f(k,x(k),u(k),v(k))                    (1)
%                     z(k) = h(k,x(k)) + w(k)                       (2)
%
% where v(k) is zero-mean, white noise with covariance Q(k) and w(k) is
% zero-mean, white noise with covariance R(k). This function implements the
% regularized particle filter detailed in algorithm 6 in
%
%   M.S. Arulampalam, S. Maskell, N. Gordon and T. Clapp, "A tutorial on
%   particle filters for online nonlinear/non-Gaussian Bayesian tracking",
%   IEEE Trans. Signal Process., vol. 50, no. 2, pp. 174-188, Feb. 2002.
%   DOI: https://doi.org/10.1109/78.978374
%
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
%   q       The function handle that samples from the proposal
%           distributtion. Suppose q = @resample. The first line of
%           resample must be in the form
%               Xknew = resample(Xk,wk)
%           where Xk are the particles at time step k and wk are their
%           weights.
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
%   NT      The threshold number of particles for resampling. If the number
%           of effective particles after dynamic propogation is less than
%           this value, they are re-sampled.
%
%   nRK     The number of intermediate Runge-Kutta integrations steps used
%           in the discretization of the nonlinear dynamics. May be given
%           as an empty array to use the default number.
%
%   params  A struct of constants that get passed to the dynamics model and
%           measurement model functions.
%  
% Outputs:
%
%   xhat    The N x nx array that contains the time history of the
%           state vector estimates.
%
%   P       The nx x nx x N array that contains the time history of the
%           estimation error covariance matrices.
%

% Check to see whether we have non-stationary noise, which may
% be prescribed by an array of matrices or a function handle that is a
% fucntion of the timestep/time.
if ~isa(R,'function_handle')
    if size(R,3) > 1, Rk = @(k) R(:,:,k); else, Rk = @(k) R; end
else
    Rk = R;
end
if ~isa(Q,'function_handle')
    if size(Q,3) > 1, Qk = @(k) Q(:,:,k); else, Qk = @(k) Q; end
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
xhat = zeros(N,nx);
P = zeros(nx,nx,N);
xhat(1,:) = xhat0.';
P(:,:,1) = P0;

% Compute constants used in the re-smapling method.
cfrakxk = unitHyperVolume(nx);
A = ((8/cfrakxk)*(nx+4)*(2*sqrt(pi))^nx)^(1/(nx+4));
hopt = A/(Ns^(1/(nx+4)));

% Initialize intermediate variables that get populated for each particle.
Xkp1 = zeros(nx,Ns);
logwtilkp1 = zeros(Ns,1);

% if no inputs, set to zero
if isempty(u)
    u = zeros(N,1);
end

% Initialize the particles by sampling Ns samples independently from
% N(xhat(0), P(0)) to generate Xi(0) for i = 1,...,Ns. Also set wi0 = 1/Ns
% for all i = 1,...,Ns.
Sx0 = chol(P0)';
Xk = xhat0 + Sx0*randn(nx,Ns);
wk = (1/Ns)*ones(Ns,1);

% This loop performs one model propagation step and one measurement
% update step per iteration.
for k = 1:N-1

    disp(k);

    % Generate process noise values for the particles by sampling vi(k)
    % independently from N(0;Q(k)) for i = 1,...,Ns. This is done in
    % preparation for dynamic propagation of the particles.
    Sv = chol(Qk(k))';
    Vk = Sv*randn(nv,Ns);

    % Dynamically propagate the particles by computing and compute the
    % corresponding natural logarithms of their un-normalized weights.
    uk = u(k,:).';
    ukp1 = u(k+1,:).';
    zkp1 = z(k+1,:).';
    for ii = 1:Ns
        if isempty(t)
            Xkp1(:,ii) = feval(f,k,Xk(:,ii),uk,Vk(:,ii),0,params);
            zbarikp1 = feval(hk,k+1,Xkp1(:,ii),ukp1,0,params);
        else
            tk = t(k,1);
            tkp1 = t(k+1,1);
            Xkp1(:,ii) = c2dNonlinear(Xk(:,ii),uk,Vk(:,ii),tk,tkp1,nRK,f,0,params);
            zbarikp1 = feval(hk,tkp1,Xkp1(:,ii),ukp1,0,params);
        end
        nui = zkp1 - zbarikp1;
        logwtilkp1(ii,1) = log(wk(ii,1)) - 0.5*(nui'/Rk(k+1))*nui;
    end

    % Find imax st log[wtil(imax)] >= log[wtil(i)] for all i = 1,...,Ns.
    logwtilkp1max = max(logwtilkp1);

    % Next, compute the modified un-normalized weights.
    wtiltilkp1 = exp(logwtilkp1-logwtilkp1max);

    % Finally, compute the normalized weights.
    wkp1 = wtiltilkp1/sum(wtiltilkp1);

    % Compute the particle approximations of the a posteriori state
    % estimate and its error covariance matrix.
    for ii = 1:Ns
        xhat(k+1,:) = xhat(k+1,:) + (wkp1(ii,1)*Xkp1(:,ii)).';
        delxi = Xkp1(:,ii) - xhat(k+1,:).';
        P(:,:,k+1) = P(:,:,k+1) + wkp1(ii,1)*(delxi*delxi');
    end    

    % Compute the effective number of particles.
    Neff = 1/sum(wkp1.^2);

    % If Neff > NT, continue to the next sample.
    if Neff > NT
        Xk = Xkp1;
        wk = wkp1;
        continue
    end

    % Otherwise, resample X(k+1) and w(k+1) for i = 1,...,Ns to
    % generate Xnew(k+1) for l = 1,...,Ns using the provided proposal
    % distribution q.
    Xnewkp1 = feval(q,Xkp1,wkp1);

    % Use Cholesky factorization in order to compute S(k+1).
    [Skp1tr,flag] = chol(P(:,:,k+1));
    Skp1 = Skp1tr';

    % If P(k+1) was not positive definite, use the eigenvalue decomposition
    if flag > 0 
        [V,D] = eig(P(:,:,k+1));
        for ii = 1:nx
            if D(ii,ii) < 0
                D(ii,ii) = 0;
            end
        end
        Skp1 = V*sqrt(D);
    end

    % Sample beta(l) independently from K(beta) for l = 1,...,Ns and
    % compute resampled Xtilnew(k+1).
    Xtilnewkp1 = zeros(nx,Ns);
    beta = epanechnikovSample(nx,Ns);
    for ll = 1:Ns
        Xtilnewkp1(:,ll) = Xnewkp1(:,ll) + hopt*Skp1*beta(:,ll);
    end
    
    % Set wnew(k+1) = 1/Ns for l = 1,...,Ns.
    wnewkp1 = (1/Ns)*ones(Ns,1);

    % Replace X(k+1) and w(k+1) for i = 1,...,Ns by Xtilnew(k+1) and
    % wnew(k+1) for l = 1,...,Ns, and iterate k.
    Xk = Xtilnewkp1;
    wk = wnewkp1;

end

end