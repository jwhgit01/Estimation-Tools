function [xhat,P] = unscentedKalmanFilterDT(t,z,u,fc,hk,Q,R,xhat0,P0,nRK,alpha,beta,kappa,params)
%unscentedKalmanFilterDT
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs square-root discrete-time unscented Kalman
% filtering for a given time history of measurments and the discrete-time
% nonlinear system,
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
%   h
%   
%   Q,R     The process and measurement noise covariance.
%
%   xhat0   The nx1 initial state estimate.
%
%   P0      The nxn symmetric positive definite initial state
%           estimation error covariance matrix.
%
%
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
    if size(Q,3) > 1, Qk = @(k) Q(:,:,k); else, Qk = @(k) Q; end
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
xhat = zeros(N,nx);
P = zeros(nx,nx,N);
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
for k = 1:N-1
    
    disp(k)

    % Compute the lower Cholesky factors of P(k) and Q(k)
    Sx = chol(P(:,:,k))';
    Sv = chol(Qk(k))';

    % Generate 1+2*nx+2*nv sigma points. Propagate these sigma points
    % through the dynamics model that transitions from sample k to sample
    % k+1 and through the measurement model at sample k+1 to compute
    % corresponding a priori sigma points for x(k+1) and z(k+1).
    xhatk = xhat(k,:).';
    uk = u(k,:).';
    Xk = zeros(nx,ns);
    Xk(:,1) = xhatk;
    Xbarkp1 = zeros(nx,ns);
    Zbarkp1 = zeros(nz,ns);
    if DT
        Xbarkp1(:,1) = feval(fc,k,xhatk,uk,zeros(nv,1),0,params);
        Zbarkp1(:,1) = feval(hk,k,xhatk,uk,0,params);
    else
        tk = t(k,1);
        tkp1 = t(k+1,1);
        Xbarkp1(:,1) = c2dNonlinear(xhatk,uk,zeros(nv,1),tk,tkp1,nRK,fc,0,params);
        Zbarkp1(:,1) = feval(hk,tk,xhatk,uk,0,params);
    end
    
    for ii = 2:(nx+1)
        Xk(:,ii) = xhatk + sqrtc*Sx(:,ii-1);
        Vk = zeros(nv,1);
        if DT
            Xbarkp1(:,ii) = feval(fc,k,Xk(:,ii),uk,Vk,0,params);
            Zbarkp1(:,ii) = feval(hk,k,Xk(:,ii),uk,0,params);
        else
            Xbarkp1(:,ii) = c2dNonlinear(Xk(:,ii),uk,Vk,tk,tkp1,nRK,fc,0,params);
            Zbarkp1(:,ii) = feval(hk,tk,Xk(:,ii),uk,0,params);
        end
    end
    for ii = (nx+2):(2*nx+1)
        Xk(:,ii) = xhatk - sqrtc*Sx(:,ii-1-nx);
        Vk = zeros(nv,1);
        if DT
            Xbarkp1(:,ii) = feval(fc,k,Xk(:,ii),uk,Vk,0,params);
            Zbarkp1(:,ii) = feval(hk,k,Xk(:,ii),uk,0,params);
        else
            Xbarkp1(:,ii) = c2dNonlinear(Xk(:,ii),uk,Vk,tk,tkp1,nRK,fc,0,params);
            Zbarkp1(:,ii) = feval(hk,tk,Xk(:,ii),uk,0,params);
        end
    end
    for ii = (2*nx+2):(2*nx+nv+1)
        Xk(:,ii) = xhatk;
        Vk = sqrtc*Sv(:,ii-1-2*nx);
        if DT
            Xbarkp1(:,ii) = feval(fc,k,Xk(:,ii),uk,Vk,0,params);
            Zbarkp1(:,ii) = feval(hk,k,Xk(:,ii),uk,0,params);
        else
            Xbarkp1(:,ii) = c2dNonlinear(Xk(:,ii),uk,Vk,tk,tkp1,nRK,fc,0,params);
            Zbarkp1(:,ii) = feval(hk,tk,Xk(:,ii),uk,0,params);
        end
    end
    for ii = (2*nx+nv+2):(2*nx+2*nv+1)
        Xk(:,ii) = xhatk;
        Vk = -sqrtc*Sv(:,ii-1-2*nx-nv);
        if DT
            Xbarkp1(:,ii) = feval(fc,k,Xk(:,ii),uk,Vk,0,params);
            Zbarkp1(:,ii) = feval(hk,k,Xk(:,ii),uk,0,params);
        else
            Xbarkp1(:,ii) = c2dNonlinear(Xk(:,ii),uk,Vk,tk,tkp1,nRK,fc,0,params);
            Zbarkp1(:,ii) = feval(hk,tk,Xk(:,ii),uk,0,params);
        end
    end

    % Compute the a priori means and covariances of the state and the
    % measurement at sample k+1 along with their cross correlation.
    xbarkp1 = Xbarkp1*Wm;
    zbarkp1 = Zbarkp1*Wm;
    Pbarkp1 = zeros(nx,nx);
    Pxzbarkp1 = zeros(nx,nz);
    Pzzbarkp1 = Rk(k+1);
    for jj = 1:ns
        Pbarkp1 = Pbarkp1 + Wc(jj,1)*((Xbarkp1(:,jj)-xbarkp1)*(Xbarkp1(:,jj)-xbarkp1)');
        Pxzbarkp1 = Pxzbarkp1 + Wc(jj,1)*((Xbarkp1(:,jj)-xbarkp1)*(Zbarkp1(:,jj)-zbarkp1)');
        Pzzbarkp1 = Pzzbarkp1 + Wc(jj,1)*((Zbarkp1(:,jj)-zbarkp1)*(Zbarkp1(:,jj)-zbarkp1)');
    end

    % Use the LMMSE equations to do the measurement update.
    nu = z(k+1,:).' - zbarkp1;
    xhat(k+1,:) = (xbarkp1 + (Pxzbarkp1/Pzzbarkp1)*nu).';
    P(:,:,k+1) = Pbarkp1 - (Pxzbarkp1/Pzzbarkp1)*Pxzbarkp1';
    
end

end