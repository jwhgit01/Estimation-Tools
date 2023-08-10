function [xhat,P] = unscentedKalmanFilterDT(t,z,u,f,h,Q,R,xhat0,P0,nRK,...
                                                   alpha,beta,kappa,params)
%unscentedKalmanFilterDT
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs discrete-time unscented Kalman filtering for a
% given time history of measurments and the discrete-time nonlinear system,
%
%                   x(k+1) = f(k,x(k),u(k),v(k))                    (1)
%                     z(k) = h(k,x(k)) + w(k)                       (2)            
%
% where v(k) is zero-mean, white noise with covariance Q(k) and w(k) is
% zero-mean, white noise with covariance R(k).
%
% Inputs:
%
%   t       The N x 1 sample time vector. If f is a discrete-time dynamic
%           model, t must be givenn as an empty array, []. The first sample
%           occurs after the initial condition at t = t0 or k = 0.
%
%   z       The N x nz time history of measurements.
%
%   u       The N x nu time history of system inputs (optional). The first
%           input occurs at t = t0 or k = 0. If not applicable set to an
%           empty array, [].
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
%   alpha   determines the spread of the sigma points about xbar. 
%           Typically, one chooses 10e-4 <= alpha <= 1.
%
%   beta    is another tuning parameter that incorporates information
%           about the prior distribution of x. The value of beta = 2 is
%           optimal for a Gaussian distribution because it optimizes some
%           type of matching of higher order terms (see Wan and van der
%           Merwe).
%
%   kappa   is a secondary scaling parameter. A good value is typically
%           3-nx-nv or 3-nx.
%
%   params  A struct of constants that get passed to the dynamics model and
%           measurement model functions.
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

% Check to see whether we have non-stationary noise, which may
% be prescribed by an array of matrices or a function handle that is a
% fucntion of the sample with the intital condition occuring at k = 0.
if ~isa(R,'function_handle')
    if size(R,3) > 1, Rk = @(k) R(:,:,k+1); else, Rk = @(k) R; end
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

% Default number of runge-kutta integration steps
if isempty(nRK)
    nRK = 10;
end

% Get the problem dimensions and initialize the output arrays.
N = size(z,1);
nx = size(xhat0,1);
nz = size(z,2);
nv = size(Qk(1),1);
ns = 2*nx + 2*nv + 1;
xhat = zeros(N+1,nx);
P = zeros(nx,nx,N+1);
xhat(1,:) = xhat0.';
P(:,:,1) = P0;

% if no inputs, set to zero
if isempty(u)
    u = zeros(N,1);
end

% Compute the weights associated with the sigma points.
lambda = alpha^2*(nx+nv+kappa)-(nx+nv);
c = nx + nv + lambda;
sqrtc = sqrt(c);
Wm = zeros(ns,1);
Wc = zeros(ns,1);
Wm(1,1) = lambda/c;
Wc(1,1) = lambda/c + (1-alpha^2+beta);
Wm(2:ns,1) = repmat(1/(2*c),ns-1,1);
Wc(2:ns,1) = repmat(1/(2*c),ns-1,1);

% This loop performs one model propagation step and one measurement
% update step per iteration.
for k = 0:N-1

    % Recall, arrays are 1-indexed, but the initial condition occurs at k=0
    kp1 = k+1;

    % Compute the lower Cholesky factors of P(k) and Q(k)
    Sx = chol(P(:,:,kp1))';
    Sv = chol(Qk(k))';

    % Generate 1+2*nx+2*nv sigma points. Propagate these sigma points
    % through the dynamics model that transitions from sample k to sample
    % k+1 and through the measurement model at sample k+1 to compute
    % corresponding a priori sigma points for x(k+1) and z(k+1).
    xhatk = xhat(kp1,:).';
    uk = u(kp1,:).';
    Xk = zeros(nx,ns);
    Xk(:,1) = xhatk;
    Xbarkp1 = zeros(nx,ns);
    Zbarkp1 = zeros(nz,ns);
    if DT
        Xbarkp1(:,1) = feval(f,k,xhatk,uk,zeros(nv,1),0,params);
        Zbarkp1(:,1) = feval(h,k,Xbarkp1(:,1),uk,0,params);
    else
        tk = t(kp1,1);
        tkp1 = t(kp1+1,1);
        Xbarkp1(:,1) = c2dNonlinear(xhatk,uk,zeros(nv,1),tk,tkp1,nRK,f,0,params);
        Zbarkp1(:,1) = feval(h,tk,Xbarkp1(:,1),uk,0,params);
    end
    for ii = 2:(nx+1)
        Xk(:,ii) = xhatk + sqrtc*Sx(:,ii-1);
        Vk = zeros(nv,1);
        if DT
            Xbarkp1(:,ii) = feval(f,k,Xk(:,ii),uk,Vk,0,params);
            Zbarkp1(:,ii) = feval(h,k,Xbarkp1(:,ii),uk,0,params);
        else
            Xbarkp1(:,ii) = c2dNonlinear(Xk(:,ii),uk,Vk,tk,tkp1,nRK,f,0,params);
            Zbarkp1(:,ii) = feval(h,tk,Xbarkp1(:,ii),uk,0,params);
        end
    end
    for ii = (nx+2):(2*nx+1)
        Xk(:,ii) = xhatk - sqrtc*Sx(:,ii-1-nx);
        Vk = zeros(nv,1);
        if DT
            Xbarkp1(:,ii) = feval(f,k,Xk(:,ii),uk,Vk,0,params);
            Zbarkp1(:,ii) = feval(h,k,Xbarkp1(:,ii),uk,0,params);
        else
            Xbarkp1(:,ii) = c2dNonlinear(Xk(:,ii),uk,Vk,tk,tkp1,nRK,f,0,params);
            Zbarkp1(:,ii) = feval(h,tk,Xbarkp1(:,ii),uk,0,params);
        end
    end
    for ii = (2*nx+2):(2*nx+nv+1)
        Xk(:,ii) = xhatk;
        Vk = sqrtc*Sv(:,ii-1-2*nx);
        if DT
            Xbarkp1(:,ii) = feval(f,k,Xk(:,ii),uk,Vk,0,params);
            Zbarkp1(:,ii) = feval(h,k,Xbarkp1(:,ii),uk,0,params);
        else
            Xbarkp1(:,ii) = c2dNonlinear(Xk(:,ii),uk,Vk,tk,tkp1,nRK,f,0,params);
            Zbarkp1(:,ii) = feval(h,tk,Xbarkp1(:,ii),uk,0,params);
        end
    end
    for ii = (2*nx+nv+2):(2*nx+2*nv+1)
        Xk(:,ii) = xhatk;
        Vk = -sqrtc*Sv(:,ii-1-2*nx-nv);
        if DT
            Xbarkp1(:,ii) = feval(f,k,Xk(:,ii),uk,Vk,0,params);
            Zbarkp1(:,ii) = feval(h,k,Xbarkp1(:,ii),uk,0,params);
        else
            Xbarkp1(:,ii) = c2dNonlinear(Xk(:,ii),uk,Vk,tk,tkp1,nRK,f,0,params);
            Zbarkp1(:,ii) = feval(h,tk,Xbarkp1(:,ii),uk,0,params);
        end
    end

    % Compute the a priori means and covariances of the state and the
    % measurement at sample k+1 along with their cross correlation.
    xbarkp1 = Xbarkp1*Wm;
    zbarkp1 = Zbarkp1*Wm;
    Pbarkp1 = zeros(nx,nx);
    Pxzbarkp1 = zeros(nx,nz);
    Pzzbarkp1 = Rk(kp1);
    for jj = 1:ns
        Pbarkp1 = Pbarkp1 + Wc(jj,1)*((Xbarkp1(:,jj)-xbarkp1)*(Xbarkp1(:,jj)-xbarkp1)');
        Pxzbarkp1 = Pxzbarkp1 + Wc(jj,1)*((Xbarkp1(:,jj)-xbarkp1)*(Zbarkp1(:,jj)-zbarkp1)');
        Pzzbarkp1 = Pzzbarkp1 + Wc(jj,1)*((Zbarkp1(:,jj)-zbarkp1)*(Zbarkp1(:,jj)-zbarkp1)');
    end

    % Use the LMMSE equations to do the measurement update.
    zkp1 = z(kp1,:).';
    nukp1 = zkp1 - zbarkp1;
    xhat(kp1+1,:) = (xbarkp1 + (Pxzbarkp1/Pzzbarkp1)*nukp1).';
    P(:,:,kp1+1) = Pbarkp1 - (Pxzbarkp1/Pzzbarkp1)*Pxzbarkp1';
    
end

end