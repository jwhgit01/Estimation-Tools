function [xhat,P,nu,epsnu,sigdig] = extendedKalmanFilterDT(z,u,f,h,Q,R,xhat0,P0)
%kalmanFilter 
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs extended Kalman filtering for a given time history
% of measurments and the discrete-time nonlinear system,
%
%   x(k+1) = f(k,x,u,v)                                             (1)
%     z(k) = h(k,x) + w(k)                                          (2)
%
% where v(k) is zero-mean Gaussian, white noise with covariance Q(k) and
% w(k) is zero-mean Gaussian, white noise with covariance R(k).
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
%   Q,R     The process and measurement noise covariance. These may be
%           specified as contant matrices, (.)x(.)xN arrays of matrices, or
%           function handles that returns a matrix given the time step k.
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

% Get the problem dimensions and initialize the output arrays.
N = size(z,1);
n = size(xhat0,1);
p = size(z,2);
xhat = zeros(N,n);
P = zeros(n,n,N);
nu = zeros(N,p);
xhat(1,:) = xhat0.';
P(:,:,1) = P0;

return
%=========================================================================%

% Check to see whether we have a time-varying or time-invariant system. A
% time-varying system may be prescribed by an array of matrices or a
% function handle that is a fucntion of the time step k. Note that
% G,Gam,H,Q,R must all be the same dimesnions and type. If an input is a
% constant matrix or a 3-dim array, make it an anonymous function.
if ~isa(F,'function_handle')
    if size(F,3) > 1, Fk = @(k) F(:,:,k); else, Fk = @(k) F; end
end
if ~isempty(G) && ~isa(G,'function_handle')
    if size(G,3) > 1, Gk = @(k) F(:,:,k); else, Gk = @(k) G; end
end
if ~isa(Gam,'function_handle')
    if size(Gam,3) > 1, Gamk = @(k) Gam(:,:,k); else, Gamk = @(k) Gam; end
end
if ~isa(H,'function_handle')
    if size(H,3) > 1, Hk = @(k) H(:,:,k); else, Hk = @(k) H; end
end
if ~isa(Q,'function_handle')
    if size(Q,3) > 1, Qk = @(k) Q(:,:,k); else, Qk = @(k) Q; end
end
if ~isa(R,'function_handle')
    if size(R,3) > 1, Rk = @(k) R(:,:,k); else, Rk = @(k) R; end
end

% This loop performs one model propagation step and one measurement
% update step per iteration.
for k = 1:N-1

    % Perform the dynamic propagation of the state estimate and the
    % covariance.
    if isempty(u) || isempty(G)
        xbarkp1 = Fk(k)*xhat(k,:).';
    else
        xbarkp1 = Fk(k)*xhat(k,:).' + Gk(k)*u(k,:).';
    end
    Pbarkp1 = Fk(k)*P(:,:,k)*Fk(k)' + Gamk(k)*Qk(k)*Gamk(k)';


    % Perform the measurement update of the state estimate and the
    % covariance.
    nu(k+1,:) = (z(k+1,:).' - Hk(k+1)*xbarkp1).';
    Skp1 = Hk(k+1)*Pbarkp1*Hk(k+1)' + Rk(k+1);
    Wkp1 = Pbarkp1*Hk(k+1)'/Skp1;
    xhat(k+1,:) = (xbarkp1 + Wkp1*nu(k+1,:).').';
    P(:,:,k+1) = Pbarkp1 - Wkp1*Skp1*Wkp1';

end

end