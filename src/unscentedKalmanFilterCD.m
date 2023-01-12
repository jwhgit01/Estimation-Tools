function [xhat,P] = unscentedKalmanFilterCD(t,z,u,f,h,Q,R,xhat0,P0,nRK,alpha,beta,kappa,params)
%unscentedKalmanFilterCD 
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs continuous-discrete unscented Kalman filtering for
% a given time history of measurments and the nonlinear system,
%
%                   dx/dt = f(t,x,u,vtil)                       (1)
%                   z(t) = h(t,x) + wtil(t)                     (2)               
%
% where vtil(k) is continuous-time, zero-mean Gaussian, white noise with
% covariance Q(t) and wtil(t) is continuous-time, zero-mean Gaussian, white
% noise with covariance R(t). This filter is implemented and referenced by
% equation number using https://doi.org/10.1109/TAC.2007.904453
%
% Inputs:
%
%   t       The (N+1) x 1 sample time vector. The first sample occurs after
%           the initial condition at t(1) = t0.
%
%   z       The N x nz time history of measurements.
%
%   u       The N x nu time history of system inputs (optional). The first
%           input occurs at t = t0. If not applicable set to an empty
%           array, [].
% 
%   f       The function handle that computes the continuous-time dynamics.
%           The first line of f must be in the form
%               [f,A,D] = nonlindyn(t,x,u,vtil,dervflag,params)
% 
%   h       The function handle that computes the modeled output of the
%           system. The first line of h must be in the form
%               [h,H] = measmodel(t,x,u,dervflag,params)
%   
%   Q       The continuous-time process noise power spectral density. It
%           may be given as a constant matrix, an ()x()xN 3-dimensional
%           array, or a function handle that is a function of time.
%
%   R       The discrete-time measurement noise covariance. It may be given
%           as a constant matrix, an ()x()xN 3-dimensional array, or a
%           function handle that is a function of the sample number, k.
%
%   xhat0   The nx x 1 initial state estimate.
%
%   P0      The nx x nx symmetric positive definite initial state
%           estimation error covariance matrix.
%
%   nRK     The number of intermediate Runge-Kutta integrations steps used
%           in the integration of the nonlinear dynamics between
%           measurements. May be given as an empty array to use the default
%           number.
%
%   alpha   A scaling parameter determines the spread of the sigma points
%           about xbar. Typically, one chooses 10e-4 <= alpha <= 1.
%
%   beta    A tuning parameter that incorporates information about the
%           prior distribution of x. The value of beta = 2 is optimal for a
%           Gaussian distribution because it optimizes some type of
%           matching of higher order terms (see Wan and van der Merwe).
%
%   kappa   A secondary scaling parameter. A good value is typically 3-nx.
%
%   params  A struct of constants that get passed to the dynamics model and
%           measurement model functions.
%  
% Outputs:
%
%   xhat    The (N+1) x nx array that contains the time history of state
%           vector estimates.
%
%   P       The nx x nx x (N+1) array that contains the time history of the
%           covariance of the estimation error.
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
    if size(Q,3) > 1, Qc = @(tk) Q(:,:,find(t>=tk,1)); else, Qc = @(t) Q; end
else
    Qc = Q;
end

% Default number of runge-kutta integration steps
if isempty(nRK)
    nRK = 10;
end

% Get the problem dimensions and initialize the output arrays.
N = size(z,1);
nx = size(xhat0,1);
nz = size(z,2);
ns = 2*nx + 1;
xhat = zeros(N+1,nx);
P = zeros(nx,nx,N+1);
xhat(1,:) = xhat0.';
P(:,:,1) = P0;

% if no inputs, set to zero
if isempty(u)
    u = zeros(N,1);
end

% Compute the weights associated with the sigma points using
% Eqs. (10) and (11).
lambda = alpha^2*(nx+kappa) - nx;
c = nx + lambda;
sqrtc = sqrt(c);
Wm = zeros(2*nx+1,1);
Wc = zeros(2*nx+1,1);
Wm(1,1) = lambda/c;
Wc(1,1) = lambda/c + (1-alpha^2+beta);
Wm(2:ns,1) = repmat(1/(2*c),2*nx,1);
Wc(2:ns,1) = repmat(1/(2*c),2*nx,1);

% This loop performs one model propagation step and one measurement
% update step per iteration.
for k = 0:N-1

    % Recall, arrays are 1-indexed, but the initial condition occurs before
    % the first sample.
    kp1 = k+1;
    
    % Propogate the sigma points through the dynamics.
    tk = t(kp1);
    tkp1 = t(kp1+1);
    xhatk = xhat(kp1,:).';
    Pk = P(:,:,kp1);
    uk = u(kp1,:).';
    [xbarkp1,Pbarkp1,Xbarkp1,Ybarkp1] = predictUKBF(xhatk,Pk,uk,Qc,tk,...
                                          tkp1,nRK,f,h,sqrtc,Wm,Wc,params);

    % Perform the measurement update of the state estimate and the
    % covariance.
    ybarkp1 = Ybarkp1*Wm;
    Skp1 = zeros(nz,nz);
    Ckp1 = zeros(nx,nz);
    for ii = 1:ns
        Skp1 = Skp1 + Wc(ii,1)*(Ybarkp1(:,ii)-ybarkp1)*(Ybarkp1(:,ii)-ybarkp1)';
        Ckp1 = Ckp1 + Wc(ii,1)*(Xbarkp1(:,ii)-xbarkp1)*(Ybarkp1(:,ii)-ybarkp1)';
    end
    Skp1 = Skp1 + Rk(kp1);
    Kkp1 = Ckp1/Skp1;
    zkp1 = z(kp1,:).'; 
    nukp1 = zkp1 - ybarkp1;
    xhat(kp1+1,:) = Xbarkp1(:,1) + Kkp1*nukp1;
    P(:,:,kp1+1) = Pbarkp1 - Kkp1*Skp1*Kkp1';
    
end

end