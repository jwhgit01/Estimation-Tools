function [xhat,P,nu,epsnu,sigdig] = extendedKalmanFilterDT(t,z,u,f,h,Q,R,xhat0,P0,nRK,params)
%extendedKalmanFilterDT
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs discrete-time extended Kalman filtering for a
% given time history of measurments and the discrete-time nonlinear system,
%
%                   x(k+1) = f(k,x(k),u(k),v(k))                    (1)
%                     z(k) = h(k,x(k)) + w(k)                       (2)
%
% where v(k) is zero-mean Gaussian, white noise with covariance Q(k) and
% w(k) is zero-mean Gaussian, white noise with covariance R(k).
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
%   nu      The Nx1 vector of innovations. The first value is zero
%           because there is no measurement update at the first sample.
%
%   sigdig  The approximate number of accurate significant decimal places
%           in the result. This is computed using the condition number of
%           the covariance of the innovations, S.
%
%   epsnu   The N x 1 vector of the normalized innovation statistic. The
%           first value is zero because there is no measurement update at
%           the first sample time.
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

    disp(k);
    
    % Perform the dynamic propagation of the state estimate and the
    % covariance.
    xhatk = xhat(k,:).';
    uk = u(k,:).';
    Pk = P(:,:,k);
    if isempty(t)
        [xbarkp1,Fk,Gamk] = feval(f,k,xhatk,uk,[],1,params);
    else
        [xbarkp1,Fk,Gamk] = c2dNonlinear(xhatk,uk,zeros(nv,1),t(k,1),t(k+1,1),nRK,f,1,params);
    end
    Pbarkp1 = Fk*Pk*Fk' + Gamk*Qk(k)*Gamk';

    % Perform the measurement update of the state estimate and the
    % covariance.
    ukp1 = u(k+1,:).';
    if isempty(t)
        [zbarkp1,Hkp1] = feval(h,k+1,xbarkp1,ukp1,1,params);
    else
        [zbarkp1,Hkp1] = feval(h,t(k+1,1),xbarkp1,ukp1,1,params);
    end
    nu(k+1,:) = (z(k+1,:).'-zbarkp1).';
    Skp1 = Hkp1*Pbarkp1*Hkp1' + Rk(k+1);
    Wkp1 = Pbarkp1*Hkp1'/Skp1;
    xhat(k+1,:) = (xbarkp1 + Wkp1*nu(k+1,:).').';
    P(:,:,k+1) = Pbarkp1-Wkp1*Skp1*Wkp1';

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