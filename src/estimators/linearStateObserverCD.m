function [xhat,yhat] = linearStateObserverCD(z,u,F,G,H,D,K,xhat0)
%linearStateObserverCD
%
% Copyright (c) 2023 Jeremy W. Hopwood. All rights reserved.
%
% This function performs linear time-varying state observation for the
% continuous-time system
%
%                        dx/dt = A(t)*x(t) + B(t)*u(t)                  (1)
%
% with discrete measurements
%
%                         y(k) = H*x(tk) + D*u(tk)                      (2)
%
% given a possibly time-varying gain matrix K(t).
%
% Inputs:
%
%   z       The N x nz time history of measurements. The first sample
%           occurs after the initial condition of k = 0;
%
%   u       The (N+1) x nu time history of system inputs (optional). The first
%           input occurs at k = 0. If not applicable set to an empty array.
% 
%   A,B    The system matrices in Eq.(1). If there is no input, B should be
%           an empty array, [].
% 
%   H,D     The measurement model matrices in Eq.(2). If there is no throughput,
%           D should be an empty array, [].
%   
%   K       The nx x ny observer gain matrix.
%
%   xhat0   The nx x 1 initial state estimate occuring at sample k=0.
%
%  
% Outputs:
%
%   xhat    The (N+1) x nx array that contains the time history of the
%           state vector estimates. The index of the estimate is 1 plus the
%           sample number (i.e. k=0 --> index=1).
%
%   yhat    The N x ny array that contains the time history of the
%           output estimates.
%

% Get the problem dimensions and initialize the output arrays.
N = size(z,1);
nx = size(xhat0,1);
ny = size(z,2);
xhat = zeros(N+1,nx);
xhat(1,:) = xhat0.';
xbar = zeros(N,nx);
yhat = zeros(N,ny);
ybar = zeros(N,ny);

% This loop performs one model propagation step and one measurement
% update step per iteration.
for k = 0:N-1

    % Recall, arrays are 1-indexed, but the initial condition occurs at k=0
    kp1 = k+1;

    % Perform the dynamic propagation of the state vector.
    xhatk = xhat(kp1,:).';
    if isempty(u) || isempty(G)
        xbarkp1 = F*xhatk;
    else
        uk = u(kp1,:).';
        xbarkp1 = F*xhatk + G*uk;
    end
    xbar(kp1,:) = xbarkp1.';
    
    % Output prediction without correction.
    if isempty(u) || isempty(D)
        ybarkp1 = H*xbarkp1;
    else
        ukp1 = u(kp1+1,:).';
        ybarkp1 = H*xbarkp1 + D*ukp1;
    end
    ybar(kp1,:) = ybarkp1.';

    % Perform the measurement update using the gain matrix, K.
    zkp1 = z(kp1,:).';
    ytildekp1 = zkp1 - ybarkp1;
    xhat(kp1+1,:) = (xbarkp1 + K*ytildekp1).';

end

end