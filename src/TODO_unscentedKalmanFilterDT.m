function [xhat,P,nu,epsnu] = unscentedKalmanFilterDT(t,z,u,f,h,Q,Rk,xhat0,P0,params)
%unscentedKalmanFilterDT 
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs square-root continuous-discrete unscented Kalman
% filtering for a given time history of measurments and the discrete-time
% nonlinear system,
%
%                   x(k+1) = f(k,x,u,v)                         (1)
%                     z(t) = h(k,x) + w(k)                      (2)
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

% number of runge-kutta integration steps
nRK = 10;

% Get the problem dimensions and initialize the output arrays.
N = size(z,1);
nx = size(xhat0,1);
nz = size(z,2);
nv = size(Q,1);
xhat = zeros(N,nx);
P = zeros(nx,nx,N);
nu = zeros(N,nz);
epsnu = zeros(N,1);
xhat(1,:) = xhat0.';
P(:,:,1) = P0;

L = ????
Wc = ????
Wm = ????
ns = 2*(nx+nv);
c = ????
sqrtc = sqrt(c);

% This loop performs one model propagation step and one measurement
% update step per iteration.
for k = 1:N-1
    
    % Compute the Cholesky factors of P(k) and Q(k).
    Sxk = chol(P(:,:,k));
    Svk = chol(Qk);

    % Perform the square-root continuous-time prediction
    Sxkinv = inv(Sxk);
    W = (eye(ns)-repmat(Wm,1,ns))*diag(Wc)*(eye(ns)-repmat(Wm,1,ns))';
    
    [~,~,D] = feval(f,t(k),xhat(k,:).',u(k),zeros(nv,1),1,params);
    M = Sxkinv'*(X*W*fX + fX*W*X' + D*Q*D')*Sxkinv;
    Phi = zeros(nx,nx);
    for ii = 1:nx
        for jj = 1:nx
            if ii>jj
                Phi(ii,jj) = M(ii,jj);
            elseif ii==jj
                Phi(ii,jj) = 0.5*M(ii,jj);
            end
        end
    end

    % Create matrix of sigma points
    Xk0 = xhat(k,:).';
    vsk0 = zeros(nv,1);
    Xki = zeros(nx,2*(nx+nv));
    vski = zeros(nv,2*(nx+nv));
    for ii = 1:nx
        Xki(:,ii) = Xk0 + sqrtc*Sxk(:,ii);
    end
    for ii = (nx+1):(2*nx)
        Xki(:,ii) = Xk0 - sqrtc*Sxk(:,ii-nx);
    end
    for ii = (2*nx+1):(2*nx+nv)
        Xki(:,ii) = Xk0;
        vski(:,ii) = sqrtc*Svk(:,ii-2*nx);
    end
    for ii = (2*nx+nv+1):(2*(nx+nv))
        Xki(:,ii) = Xk0;
        vski(:,ii) = -sqrtc*Svk(:,ii-2*nx-nv);
    end




    APM = sqrtc*[zeros(nx,1), Sxk'*Phi, -Sxk'*Phi];
    Xkp1 = zeros(nx,ns);
    for ii = 1:ns
        fXfun = @(x,u,v,t,tkp1,nRK,f,flag,params) fX*Wm+APM(:,ii);
        Xkp1(:,ii) = c2dNonlinear(Xk(:,ii),u(k),zeros(nv,1),t(k),t(k+1),nRK,fXfun,0,params);
    end
    

    % Perform the dynamic propagation of the state estimate and the
    % covariance.
    xhatk = xhat(k,:).';
    xbarkp1 = c2dNonlinear(xhatk,u(k),zeros(nv,1),t(k),t(k+1),nRK,f,0,params);
    [~,A,~] = feval(f,t(k),xhatk,u(k),zeros(nv,1),1,params);
    Pk = P(:,:,k);
    Pbarkp1 = c2dNonlinear(Pk(:),A,Q,t(k),t(k+1),nRK,@covarianceDynamicsCT,0,params);
    Pbarkp1 = reshape(Pbarkp1,nx,nx);

    % Perform the measurement update of the state estimate and the
    % covariance.
    [hkp1,Hkp1,~] = feval(h,t(k+1),xbarkp1,1,0);
    nu(k+1,:) = (z(k+1,:).' - hkp1).';
    Skp1 = Hkp1*Pbarkp1*Hkp1' + Rk;
    Wkp1 = Pbarkp1*Hkp1'/Skp1;
    xhat(k+1,:) = (xbarkp1 + Wkp1*nu(k+1,:).').';
    P(:,:,k+1) = (eye(nx)-Wkp1*Hkp1)*Pbarkp1;

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