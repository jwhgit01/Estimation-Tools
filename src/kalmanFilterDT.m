function [xhat,P,nu,epsnu,sigdig] = kalmanFilterDT(z,u,F,G,Gam,H,Q,R,xhat0,P0)
%kalmanFilterDT
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs linear Kalman filtering for a given time history
% of measurments and the discrete-time linear system,
%
%           x(k+1) = F(k)*x(k) + G(k)*u(k) + Gam(k)*v(k)            (1)
%             z(k) = H(k)*x(k) + w(k)                               (2)
%
% where v(k) is zero-mean Gaussian, white noise with covariance Q(k) and
% w(k) is zero-mean Gaussian, white noise with covariance R(k).
%
% Inputs:
%
%   z       The N x nz time history of measurements. The first sample
%           occurs after the initial condition of k = 0;
%
%   u       The N x nu time history of system inputs (optional). The first
%           input occurs at k = 0. If not applicable set to an empty array.
% 
%   F,G,Gam The system matrices in Eq.(1). These may be specified as
%           contant matrices, (.)x(.)xN arrays of matrices, or function
%           handles that return a matrix given the sample k. Note that
%           if there is no input, G may be given as an empty array, [].
% 
%   H       The measurement model matrix in Eq.(2). This may be specified
%           as a contant matrix, an (.)x(.)xN array, or a function handle
%           that returns a matrix given the sample k.
%   
%   Q,R     The process and measurement noise covariance. These may be
%           specified as contant matrices, (.)x(.)xN arrays of matrices, or
%           function handles that returns a matrix given the time step k.
%
%   xhat0   The nx x 1 initial state estimate occuring at sample k=0.
%
%   P0      The nx x nx symmetric positive definite initial state
%           estimation error covariance matrix.
%  
% Outputs:
%
%   xhat    The (N+1) x nx array that contains the time history of the
%           state vector estimates. The index of the estimate is 1 plus the
%           sample number (i.e. k=0 --> index=1).
%
%   P       The nx x nx x (N+1) array that contains the time history of the
%           estimation error covariance matrices.
%
%   nu      The (N+1) x nz vector of innovations. The first value is zero
%           because there is no measurement update at the first sample.
%
%   epsnu   The (N+1) x 1 vector of the normalized innovation statistic. The
%           first value is zero because there is no measurement update at
%           the first sample time.
%
%   sigdig  The approximate number of accurate significant decimal places
%           in the result. This is computed using the condition number of
%           the covariance of the innovations, S.
%

% Get the problem dimensions and initialize the output arrays.
N = size(z,1);
n = size(xhat0,1);
p = size(z,2);
xhat = zeros(N+1,n);
P = zeros(n,n,N+1);
nu = zeros(N+1,p);
epsnu = zeros(N+1,1);
xhat(1,:) = xhat0.';
P(:,:,1) = P0;
maxsigdig = -fix(log10(eps));
sigdig = maxsigdig;

% Check to see whether we have a stationary or non-stationary system. A
% non-stationary system may be prescribed by an array of matrices or a
% function handle that is a fucntion of the time step k with the intital
% sample being k = 1. If an input is a constant matrix or a 3-dim array,
% make it an anonymous function.
if ~isa(F,'function_handle')
    if size(F,3) > 1, Fk = @(k) F(:,:,k+1); else, Fk = @(k) F; end
end
if ~isempty(G) && ~isa(G,'function_handle')
    if size(G,3) > 1, Gk = @(k) F(:,:,k+1); else, Gk = @(k) G; end
end
if ~isa(Gam,'function_handle')
    if size(Gam,3) > 1, Gamk = @(k) Gam(:,:,k+1); else, Gamk = @(k) Gam;end
end
if ~isa(H,'function_handle')
    if size(H,3) > 1, Hk = @(k) H(:,:,k+1); else, Hk = @(k) H; end
end
if ~isa(Q,'function_handle')
    if size(Q,3) > 1, Qk = @(k) Q(:,:,k+1); else, Qk = @(k) Q; end
end
if ~isa(R,'function_handle')
    if size(R,3) > 1, Rk = @(k) R(:,:,k+1); else, Rk = @(k) R; end
end

% This loop performs one model propagation step and one measurement
% update step per iteration.
for k = 0:N-1

    % Recall, arrays are 1-indexed, but the initial condition occurs at k=0
    kp1 = k+1;

    % Perform the dynamic propagation of the state estimate and the
    % covariance.
    xhatk = xhat(kp1,:).';
    if isempty(u) || isempty(G)
        xbarkp1 = Fk(k)*xhatk;
    else
        uk = u(kp1,:).';
        xbarkp1 = Fk(k)*xhatk + Gk(k)*uk;
    end
    Pk = P(:,:,kp1);
    Pbarkp1 = Fk(k)*Pk*Fk(k)' + Gamk(k)*Qk(k)*Gamk(k)';

    % Perform the measurement update of the state estimate and the
    % covariance.
    zkp1 = z(kp1,:).';
    nu(kp1+1,:) = (zkp1 - Hk(kp1)*xbarkp1).';
    Skp1 = Hk(k+1)*Pbarkp1*Hk(k+1)' + Rk(k+1);
    Wkp1 = (Pbarkp1*Hk(k+1)')/Skp1;
    xhat(kp1+1,:) = (xbarkp1 + Wkp1*nu(kp1+1,:).').';
    P(:,:,kp1+1) = Pbarkp1 - Wkp1*Skp1*Wkp1';

    % Check the condition number of Skp1 and infer the approximate accuracy
    % of the resulting estimate.
    sigdigkp1 = maxsigdig-fix(log10(cond(Skp1)));
    if sigdigkp1 < sigdig
        sigdig = sigdigkp1;
    end

    % Compute the innovation statistic, epsilon_nu(k).
    epsnu(kp1+1) = nu(kp1+1,:)*(Skp1\nu(kp1+1,:).');

end

end