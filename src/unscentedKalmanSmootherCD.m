function [xs,Ps,xhat,Phat] = unscentedKalmanSmootherCD(t,z,u,f,h,Q,R,...
                                      xhat0,P0,nRK,alpha,beta,kappa,params)
%unscentedKalmanSmootherCD 
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs continuous–discrete-time unscented
% Rauch–Tung–Striebel smoothing given a time history of measurements and
% the dynamical system
%
%                   dx/dt = f(t,x,u,vtil)                       (1)
%                   z(t) = h(t,x) + wtil(t)                     (2)
%
% where vtil(k) is continuous-time, zero-mean Gaussian, white noise with
% covariance Q(t) and wtil(t) is continuous-time, zero-mean Gaussian, white
% noise with covariance R(t).
%
% Inputs:
%
%   t       The N x 1 sample time vector. The first sample occurs after the
%           initial condition at t = t0.
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
%   xs      The (N+1) x nx array that contains the time history of smoothed
%           state vector estimates.
%
%   Ps      The nx x nx x (N+1) array that contains the time history of the
%           covariance of the smoothed estimation error.
%
%   xhat    The (N+1) x nx array that contains the time history of state
%           vector estimates before smoothing.
%
%   Phat    The nx x nx x (N+1) array that contains the time history of the
%           covariance of the estimation error before smoothing.
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

% Get the problem dimensions.
N = size(z,1);
nx = size(xhat0,1);

% if no inputs, set to zero
if isempty(u)
    u = zeros(N,1);
end

% First, perform unscented Kalman filtering forward in time.
[xhat,Phat] = unscentedKalmanFilterCD(t,z,u,f,h,Qc,Rk,xhat0,P0,nRK,alpha,beta,kappa,params);

% Initialize the outputs
xs = xhat;
Ps = Phat;

% Compute the weights associated with the sigma points.
lambda = alpha^2*(nx+kappa) - nx;
c = nx + lambda;
sqrtc = sqrt(c);
Wm = zeros(2*nx+1,1);
Wc = zeros(2*nx+1,1);
Wm(1,1) = lambda/c;
Wc(1,1) = lambda/c + (1-alpha^2+beta);
Wm(2:(2*nx+1),1) = repmat(1/(2*c),2*nx,1);
Wc(2:(2*nx+1),1) = repmat(1/(2*c),2*nx,1);

% Get the covariance and mean at the last sample.
Pskm1 = Phat(:,:,N+1);
xskm1 = xhat(N+1,:).';

% This loop propagates backwards in time and performs RTS smoothing.
for k = N:-1:1

    % Recall, arrays are 1-indexed, but the initial condition occurs at k=0
    kp1 = k+1;

    % Smooth the sigma points backwards in time.
    tk = t(kp1);
    tkm1 = t(k);
    xhatk = xhat(kp1,:).';
    Phatk = Phat(:,:,kp1);
    uk = u(k,:).';
    [xskm1,Pskm1] = smoothUKBF(xskm1,Pskm1,xhatk,Phatk,uk,Qc,tk,tkm1,...
                                               nRK,f,h,sqrtc,Wm,Wc,params);

    % Store the mean and covariance
    xs(k,:) = xskm1;
    Ps(:,:,k) = Pskm1;

end

end