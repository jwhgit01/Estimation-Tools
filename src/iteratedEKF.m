function [xhat,P] = iteratedEKF(t,z,u,f,h,Q,R,xhat0,P0,nRK,params)
%iteratedEKF 
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs ierated extended Kalman filtering for a given time
% history of measurments and the discrete-time nonlinear system,
%
%                   x(k+1) = f(k,x(k),u(k),v(k))                    (1)
%                     z(k) = h(k,x(k)) + w(k)                       (2)
%
% where v(k) is zero-mean Gaussian, white noise with covariance Q(k) and
% w(k) is zero-mean Gaussian, white noise with covariance R(k).
%
% Inputs:
%
%   t       The (N+1) x 1 sample time vector. If f is a discrete-time
%           dynamic model, t must be given as an empty array, []. The first
%           element of t corresponds to the initial condition time at
%           sample k=0.
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
%   Q       The discrete-time process noise covariance. It may be
%           specificed as a constant matrix, a ()x()xN 3-dimensional
%           array, or a function handle that is a function of the sample
%           number, k. Recall, k=0 corresponds to t=0.
%
%   R       The discrete-time measurement noise covariance of w. It may be
%           specificed as a constant matrix, a ()x()xN 3-dimensional
%           array, or a function handle that is a function of the sample
%           number, k. Recall, k=0 corresponds to t=0.
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
    if size(Q,3) > 1, Qk = @(k) Q(:,:,k+1); else, Qk = @(k) Q; end
else
    Qk = Q;
end

% Check to see whether f is a difference or differential equation.
if isempty(t)
    DT = true;
else
    DT = false;
end

% number of runge-kutta integration steps
if isempty(nRK)
    nRK = 10;
end

% Get the problem dimensions and initialize the output arrays.
N = size(z,1);
nx = size(xhat0,1);
nv = size(Qk(1),1);
xhat = zeros(N+1,nx);
P = zeros(nx,nx,N+1);
xhat(1,:) = xhat0.';
P(:,:,1) = P0;

% if no inputs, set to zero
if isempty(u)
    u = zeros(N,1);
end

% This loop performs one model propagation step and one measurement
% update step per iteration.
for k = 0:N-1

    % Recall, arrays are 1-indexed, but the initial condition occurs at k=0
    kp1 = k+1;
    
    % Measurement model used in the Gauss-Newton iterations.
    ukp1 = u(kp1,:).';
    if DT
        tkp1 = [];
    else
        tk = t(kp1,1);
        tkp1 = t(kp1+1,1);
    end
    modelGN = @(tk,xak,uk,ideriv,params) ...
        modelBSEKF(tk,tkp1,xak(1:nx,1),xak(nx+1:nx+nv,1),...
        uk,ukp1,ideriv,params,f,h,nRK);

    % Perform Gauss-Newton iterations
    xhatk = xhat(kp1,:);
    zkp1 = z(kp1,:);
    zak = [xhatk, zeros(1,nv), zkp1]; % Augmented "state"
    Pk = P(:,:,kp1);
    Rak = blkdiag(Pk,Qk(k),Rk(kp1)); % Augmented covariance
    xaguess = [xhat(kp1,:).';zeros(nv,1)];
    ukhist = u(kp1,:);
    if DT
        [xask,~,~,termflag] = gaussNewtonEstimation(k,zak,ukhist,Rak,modelGN,xaguess,1,0,params);
    else
        [xask,~,~,termflag] = gaussNewtonEstimation(tk,zak,ukhist,Rak,modelGN,xaguess,1,0,params);
    end
    xsk = xask(1:nx,1);
    vsk = xask(nx+1:nx+nv);

    % Compute futute estimate using smoothed past estimates.
    uk = ukhist.';
    if DT
        [xhatkp1,Fk,Gamk] = feval(f,k,xsk,uk,vsk,1,params);
    else
        [xhatkp1,Fk,Gamk] = c2dNonlinear(xsk,uk,vsk,tk,tkp1,nRK,f,1,params);
    end
    xhat(kp1+1,:) = xhatkp1.';

    % Compute the covariance of the estimate
    [~,Hskp1] = feval(h,tkp1,xhatkp1,ukp1,1,params);
    Pbarkp1 = Fk*Pk*Fk' + Gamk*Qk(k)*Gamk';
    P(:,:,kp1+1) = inv(inv(Pbarkp1) + (Hskp1'/Rk(kp1))*Hskp1);

end

end