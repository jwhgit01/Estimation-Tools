function [xhat,P,nu,epsnu,sigdig] = unscentedKalmanFilterCD(t,z,u,f,h,Q,R,xhat0,P0,nRK,alpha,beta,kappa,params)
%unscentedKalmanFilterCD 
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs square-root continuous-discrete unscented Kalman
% filtering for a given time history of measurments and the discrete-time
% nonlinear system,
%
%                   dx/dt = f(t,x,u) + D(t)*vtil(t)             (4)
%                    z(t) = h(t,x) + wtil(t)                    
%
% where vtil(k) is zero-mean Gaussian, white noise with covariance Q(k) and
% wtil(k) is zero-mean Gaussian, white noise with covariance R(k). This
% filter is implemented and referenced by equation number using
% https://doi.org/10.1109/TAC.2007.904453
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
%   Qc,R     The process and measurement noise covariance.
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

% Default number of runge-kutta integration steps
if isempty(nRK)
    nRK = 10;
end

% Get the problem dimensions and initialize the output arrays.
N = size(z,1);
nx = size(xhat0,1);
nz = size(z,2);
ns = 2*nx + 1;
xhat = zeros(N,nx);
X = zeros(nx,ns,N);
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

% Compute the weights associated with the sigma points using
% Eqs. (10) and (11).
lambda = alpha^2*(nx+kappa) - nx;
c = nx + lambda;
sqrtc = sqrt(c);
Wm = zeros(2*nx+1,1);
Wc = zeros(2*nx+1,1);
Wm(1,1) = lambda/c;
Wc(1,1) = lambda/c + (1-alpha^2+beta);
Wm(2:(2*nx+1),1) = repmat(1/(2*c),2*nx,1);
Wc(2:(2*nx+1),1) = repmat(1/(2*c),2*nx,1);

% This loop performs one model propagation step and one measurement
% update step per iteration.
for k = 1:N-1
    
    % Propogate the sigma points through the dynamics using the square root
    % approach.
    [xbarkp1,Pbarkp1,Xbarkp1,Ybarkp1] = predictUKBF(xhat(k,:).',P(:,:,k),u(k,:).',Qc,t(k),t(k+1),nRK,f,h,sqrtc,Wm,Wc,params);

    % Perform the measurement update of the state estimate and the
    % covariance.
    ybarkp1 = Ybarkp1*Wm;
    Skp1 = zeros(nz,nz);
    Ckp1 = zeros(nx,nz);
    for ii = 1:ns
        Skp1 = Skp1 + Wc(ii,1)*(Ybarkp1(:,ii)-ybarkp1)*(Ybarkp1(:,ii)-ybarkp1)';
        Ckp1 = Ckp1 + Wc(ii,1)*(Xbarkp1(:,ii)-xbarkp1)*(Ybarkp1(:,ii)-ybarkp1)';
    end
    Skp1 = Skp1 + Rk(k+1);
    Kkp1 = Ckp1/Skp1;
    nu(k+1,:) = z(k+1,:) - ybarkp1.';
    xhat(k+1,:) = Xbarkp1(:,1) + Kkp1*nu(k+1,:).';
    P(:,:,k+1) = Pbarkp1 - Kkp1*Skp1*Kkp1';
    
end

end