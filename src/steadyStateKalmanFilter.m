function [xhat,P,Pbar,W,nu,epsnu] = steadyStateKalmanFilter(z,u,F,G,Gam,H,Q,R,xhat0)
%kalmanFilter 
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs linear Kalman filtering for a given time history
% of measurments and the stationary discrete-time linear system,
%
%   x(k+1) = F*x(k) + G*u(k) + Gam*v(k)                             (1)
%     z(k) = H*x(k) + w(k)                                          (2)
%
% where v(k) is zero-mean Gaussian, white noise with constant covariance Q
% and w(k) is zero-mean Gaussian, white noise with constant covariance R.
%
% Inputs:
%
%   z       The Nxp time history of measurements.
%
%   u       The Nxm time history of system inputs (optional). If not
%           applicable set to an empty array, [].
% 
%   F,G,Gam The system matrices in Eq.(1).
% 
%   H       The measurement model matrix in Eq.(2).
%   
%   Q,R     The process and measurement noise covariance.
%
%   xhat0   The nx1 initial state estimate.
%
%  
% Outputs:
%
%   xhat    The Nxn array that contains the time history of the
%           state vector estimates.
%
%   P       The nxn steady-state estimation error covariance matrix.
%
%   W       The pxn steady-state Kalman gain matrix.
%
%   nu      The Nx1 vector of innovations. The first value is zero
%           because there is no measurement update at the first sample.
%
%   epsnu   The Nx1 vector of the normalized innovation statistic. The
%           first value is zero because there is no measurement update at
%           the first sample time.
%

% Get the problem dimensions and initialize the output arrays.
N = size(z,1);
n = size(xhat0,1);
p = size(z,2);
xhat = zeros(N,n);
nu = zeros(N,p);
epsnu = zeros(N,1);
xhat(1,:) = xhat0.';

% Solve the discrete-time Ricatti equation.
[Pbar,Wtr,~] = idare(F',H',Gam*Q*Gam',R,[],[]);
W = Wtr';

% Compute the a posteriori steady-state covariance.
P = (eye(n)-W*H)*Pbar*(eye(n)-W*H)' + W*R*W';

% Compute the covariance of the innovations.
S = H*Pbar*H' + R;

% This loop performs one model propagation step and one measurement
% update step per iteration.
for k = 1:N-1

    % Perform the dynamic propagation of the state estimate and the
    % covariance.
    if isempty(u) || isempty(G)
        xbarkp1 = F*xhat(k,:).';
    else
        xbarkp1 = F*xhat(k,:).' + G*u(k,:).';
    end

    % Perform the measurement update of the state estimate and the
    % covariance.
    nu(k+1,:) = (z(k+1,:).' - H*xbarkp1).';
    xhat(k+1,:) = (xbarkp1 + W*nu(k+1,:).').';

    % Compute the innovation statistic, epsilon_nu(k).
    epsnu(k+1) = nu(k+1,:)*(S\nu(k+1,:).');

end

end