function [xs,Ps,xhat,Phat] = unscentedKalmanSmootherCD(t,z,u,f,h,Q,R,xhat0,P0,nRK,alpha,beta,kappa,params)
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
%   z       The Nxp time history of measurements.
%
%   u       The Nxm time history of system inputs (optional). If not
%           applicable set to an empty array, [].
% 
%   f
% 
%   h
%   
%   Q,Rk     The process and measurement noise covariance.
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

% Get the problem dimensions.
N = size(z,1);
nx = size(xhat0,1);

% if no inputs, set to zero
if isempty(u)
    u = zeros(N,1);
end

% First, perform unscented Kalman filtering forward in time.
[xhat,Phat,~,~,~] = unscentedKalmanFilterCD(t,z,u,f,h,Qc,Rk,xhat0,P0,nRK,alpha,beta,kappa,params);

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
Pskm1 = Phat(:,:,N);
xskm1 = xhat(N,:).';

% This loop propagates backwards in time and performs RTS smoothing.
for k = N:-1:2

    % Smooth the sigma points backwards in time.
    [xskm1,Pskm1] = smoothUKBF(xskm1,Pskm1,xhat(k,:).',Phat(:,:,k),u(k,:).',Qc,t(k),t(k-1),nRK,f,h,sqrtc,Wm,Wc,params);

    % Store the mean and covariance
    xs(k-1,:) = xskm1;
    Ps(:,:,k-1) = Pskm1;

    disp(num2str(k))

end

end