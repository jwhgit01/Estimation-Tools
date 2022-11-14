function [xhat,P,nu,epsnu,sigdig] = extendedKalmanFilterCD(t,z,u,fc,h,Q,R,xhat0,P0,nRK,params)
%extendedKalmanFilterCD 
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs extended Kalman filtering for a given time history
% of measurments and the discrete-time nonlinear system,
%
%    dx/dt = f(t,x,u,vtil)                                          (1)
%     z(t) = h(t,x) + wtil(t)                                       (2)
%
% where vtil(k) is zero-mean Gaussian, white noise with covariance Q(k) and
% wtil(k) is zero-mean Gaussian, white noise with covariance R(k).
%
% Inputs:
%
%   z       The Nxp time history of measurements.
%
%   u       The Nxm time history of system inputs (optional). If not
%           applicable set to an empty array, [].
% 
%   f
% 
%   h
%   
%   Q,R     The process and measurement noise covariance.
%
%   xhat0   The nx1 initial state estimate.
%
%   P0      The nxn symmetric positive definite initial state
%           estimation error covariance matrix.
%  
% Outputs:
%
%   xhat    The (N+1)xn array that contains the time history of the
%           state vector estimates.
%
%   P       The nxnx(N+1) array that contains the time history of the
%           estimation error covariance matrices.
%
%   nu      The (N+1)x1 vector of innovations. The first value is zero
%           because there is no measurement update at the first sample.
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
xhat = zeros(N,nx);
P = zeros(nx,nx,N);
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
for k = 1:N-1

    % Perform the dynamic propagation of the state estimate and the
    % covariance.
    xhatk = xhat(k,:).';
    uk = u(k,:).';
    Pk = P(:,:,k);
    [xbarkp1,Pbarkp1] = predictEKBF(xhatk,uk,Pk,Qc,t(k),t(k+1),nRK,fc,params);

    % Perform the measurement update of the state estimate and the
    % covariance.
    [hkp1,Hkp1,~] = feval(h,t(k+1),xbarkp1,1,0);
    nu(k+1,:) = (z(k+1,:).' - hkp1).';
    Skp1 = Hkp1*Pbarkp1*Hkp1' + Rk(k+1);
    Wkp1 = Pbarkp1*Hkp1'/Skp1;
    xhat(k+1,:) = (xbarkp1 + Wkp1*nu(k+1,:).').';
    P(:,:,k+1) = (eye(nx)-Wkp1*Hkp1)*Pbarkp1;

    % Check the condition number of Skp1 and infer the approximate accuracy
    % of the resulting estimate.
    sigdigkp1 = maxsigdig-fix(log10(cond(Skp1)));
    if sigdigkp1 < sigdig
        sigdig = sigdigkp1;
    end

    % Compute the innovation statistic, epsilon_nu(k).
    epsnu(k+1) = nu(k+1,:)*(Skp1\nu(k+1,:).');

end

end