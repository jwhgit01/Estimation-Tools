function [xhat,P,nu,epsnu,sigdig] = extendedKalmanFilterCD(t,z,u,fc,h,Q,R,xhat0,P0,nRK,params)
%extendedKalmanFilterCD 
%
% Copyright (c) 2023 Jeremy W. Hopwood. All rights reserved.
%
% This function performs continuous-discrete (sometimes called hybrid)
% extended Kalman filtering for a given time history of measurments and the
% continuosu-time nonlinear system,
%
%                           dx/dt = f(t,x,u,vtil)                   (1)
%
% with discrete measurements
%
%                           z(tk) = h(tk,xk) + w(k)                 (2)
%
% where vtil(k) is continuous-time zero-mean Gaussian, white noise with
% power spectral density Q(t) and w(k) is discrete-time zero-mean Gaussian,
% white noise with covariance R(k).
%
% Inputs:
%
%   t       The (N+1) x 1 sample time vector. The first element of t 
%           corresponds to the initial condition occuring before the first
%           measurement sample at t(k), k=1.
%
%   z       The N x nz time history of measurements.
%
%   u       The (N+1) x nu time history of system inputs (optional). If not
%           applicable set to an empty array, [].
% 
%   f       The function handle that computes the continuous-time dynamics
%           of the system. The first line of f must be in the form
%               [f,A,D] = nonlindyn(t,x,u,vtil,dervflag,params)
% 
%   h       The function handle that computes the modeled output of the
%           system. The first line of h must be in the form
%               [h,H] = measmodel(t,x,u,dervflag,params)
%   
%   Q       The power spectral density of the continuous-time process noise
%           vtil. It may be specificed a a constant matrix, a ()x()xN
%           3-dimensional array, or a function handle this is a function of
%           time, t. If it is a ()x()xN array, then the first element
%           corresponds to t=0.
%
%   R       The discrete-time measurement noise covariance of w. It may be
%           specificed as a constant matrix, a ()x()xN 3-dimensional array,
%           or a function handle that is a function of the sample number,k.
%           Recall, k=0 corresponds to t=0. If it is a ()x()xN array, then
%           the first element corresponds to k=1.
%
%   xhat0   The nx x 1 initial state estimate.
%
%   P0      The nx x nx symmetric positive definite initial state
%           estimation error covariance matrix.
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
%   xhat    The (N+1) x nx array that contains the time history of the
%           state vector estimates.
%
%   P       The nx x nx x (N+1) array that contains the time history of the
%           estimation error covariance matrices.
%
%   nu      The N x nz vector of innovations.
%
%   epsnu   The N x 1 vector of the normalized innovation statistic.
%
%   sigdig  The approximate number of accurate significant decimal places
%           in the result. This is computed using the condition number of
%           the covariance of the innovations, S.
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
    if size(Q,3) > 1, Qc = @(tk) Q(:,:,find(t>=tk,1)); else, Qc = @(t) Q; end
else
    Qc = Q;
end

% number of runge-kutta integration steps
if isempty(nRK)
    nRK = 10;
end

% Get the problem dimensions and initialize the output arrays.
N = size(z,1);
nx = size(xhat0,1);
nz = size(z,2);
xhat = zeros(N+1,nx);
P = zeros(nx,nx,N+1);
nu = zeros(N,nz);
epsnu = zeros(N,1);
xhat(1,:) = xhat0.';
P(:,:,1) = P0;
maxsigdig = -fix(log10(eps));
sigdig = maxsigdig;

% if no inputs, set to zero
if isempty(u)
    u = zeros(N,1);
end

% This loop performs one model propagation step and one measurement
% update step per iteration.
for k = 0:N-1

    % Recall, arrays are 1-indexed, but the initial condition occurs at k=0
    % disp(['k = ' num2str(k)])
    kp1 = k+1;

    % Perform the dynamic propagation of the state estimate and the
    % covariance.
    xhatk = xhat(kp1,:).';
    uk = u(kp1,:).';
    Pk = P(:,:,kp1);
    tk = t(kp1,1);
    tkp1 = t(kp1+1,1);
    [xbarkp1,Pbarkp1] = predictEKBF(xhatk,uk,Pk,Qc,tk,tkp1,nRK,fc,params);

    % Perform the measurement update of the state estimate and the
    % covariance.
    zkp1 = z(kp1,:).';
    ukp1 = u(kp1,:).';
    [zbarkp1,Hkp1] = feval(h,tkp1,xbarkp1,ukp1,1,params);
    nu(kp1,:) = (zkp1-zbarkp1).';
    Skp1 = Hkp1*Pbarkp1*Hkp1' + Rk(kp1);
    Wkp1 = Pbarkp1*Hkp1'/Skp1;
    xhat(kp1+1,:) = (xbarkp1 + Wkp1*nu(kp1,:).').';
    P(:,:,kp1+1) = (eye(nx)-Wkp1*Hkp1)*Pbarkp1;

    % Check the condition number of Skp1 and infer the approximate accuracy
    % of the resulting estimate.
    sigdigkp1 = maxsigdig-fix(log10(cond(Skp1)));
    if sigdigkp1 < sigdig
        sigdig = sigdigkp1;
    end

    % Compute the innovation statistic, epsilon_nu(k).
    epsnu(kp1) = nu(kp1,:)*(Skp1\nu(kp1,:).');

end

end