function [xhat,P,nu] = sqrtUnscentedKalmanFilterCD(t,z,u,f,h,Qc,R,xhat0,P0,alpha,beta,kappa,params)
%sqrtUnscentedKalmanFilterCD 
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

% Check to see whether we have non-stationary measurement noise, which may
% be prescribed by an array of matrices or a function handle that is a
% fucntion of the time step k.
if ~isa(R,'function_handle')
    if size(R,3) > 1, Rk = @(k) R(:,:,k); else, Rk = @(k) R; end
else
    Rk = R;
end

% number of runge-kutta integration steps
nRK = 10;

% Get the problem dimensions and initialize the output arrays.
N = size(z,1);
nx = size(xhat0,1);
nz = size(z,2);
ns = 2*nx + 1;
xhat = zeros(N,nx);
X = zeros(nx,ns,N);
P = zeros(nx,nx,N);
nu = zeros(N,nz);
xhat(1,:) = xhat0.';
P(:,:,1) = P0;

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

% Compute lower triangular Cholesky factor of Pk satisfying Eq. (15).
Sx0 = chol(P0)';

% Compute the sigma points for the initial condition
[Xk0,~,~] = sigmaPointsCT(t(1),xhat(1,:).',u(1,:).',sqrtc,Sx0,f,h,params);
X(:,:,1) = Xk0;

% This loop performs one model propagation step and one measurement
% update step per iteration.
for k = 1:N-1
    
    % Propogate the sigma points through the dynamics using the square root
    % approach.
    [Xbarkp1,Pbarkp1,Ybarkp1] = predictSqrtUKBF(X(:,:,k),P(:,:,k),u(k,:).',Qc,t(k),t(k+1),nRK,f,h,sqrtc,Wm,Wc,params);
    
    % Compute lower triangular Cholesky factors of Pk and Rk.
    Sbarxkp1 = chol(Pbarkp1)';
    Sbarvkp1 = chol(Rk(k+1))';

    % Perform the square-root measurement update following:
    %   Van Der Merwe, Rudolph. Sigma-point Kalman filters for
    %   probabilistic inference in dynamic state-space models. Oregon
    %   Health & Science University, 2004.
    xbarkp1 = Xbarkp1*Wm;
    ybarkp1 = Ybarkp1*Wm;
    Sybarkp1tr = qr([sqrt(Wc(2,1))*(Ybarkp1(:,2:ns)-ybarkp1) Sbarvkp1].');
    Sybarkp1tr = Sybarkp1tr(1:nz,1:nz);
    Sybarkp1 = cholupdate(Sybarkp1tr,sqrt(abs(Wc(1,1)))*(Ybarkp1(:,1)-ybarkp1))';
    Pxykp1 = zeros(nx,nz);
    for ii = 1:ns
        Pxykp1 = Pxykp1 + Wc(ii)*(Xbarkp1(:,ii)-xbarkp1)*(Ybarkp1(:,ii)-ybarkp1).';
    end
    K = (Pxykp1/Sybarkp1')/Sybarkp1;
    nu(k+1,:) = z(k+1,:) - ybarkp1.';
    xhat(k+1,:) = xbarkp1 + K*nu(k+1,:).';
    U = K*Sybarkp1;
    Sxkp1 = cholupdate(Sbarxkp1',U,'-')';
    P(:,:,k+1) = Sxkp1*Sxkp1';

    % OR
%     % Perform the measurement update of the state estimate and the
%     % covariance.
%     xbarkp1 = Xbarkp1*Wm;
%     ybarkp1 = Ybarkp1*Wm;
%     Skp1 = zeros(nz,nz);
%     Ckp1 = zeros(nx,nz);
%     for ii = 1:ns
%         Skp1 = Skp1 + Wc(ii,1)*(Ybarkp1(:,ii)-ybarkp1)*(Ybarkp1(:,ii)-ybarkp1)';
%         Ckp1 = Ckp1 + Wc(ii,1)*(Xbarkp1(:,ii)-xbarkp1)*(Ybarkp1(:,ii)-ybarkp1)';
%     end
%     Skp1 = Skp1 + Rk(k+1);
%     Kkp1 = Ckp1/Skp1;
%     nu(k+1,:) = z(k+1,:) - ybarkp1.';
%     xhat(k+1,:) = Xbarkp1(:,1) + Kkp1*nu(k+1,:).';
%     P(:,:,k+1) = Pbarkp1 - Kkp1*Skp1*Kkp1';
    
end

end