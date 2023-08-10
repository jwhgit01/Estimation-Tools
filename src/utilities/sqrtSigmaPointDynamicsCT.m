function [dXdt,dPdt] = sqrtSigmaPointDynamicsCT(t,X,P,u,f,h,Qc,sqrtc,Wm,Wc,params)
%sqrtSigmaPointDynamicsCT
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function computes the dynamics of the continuous-time, square root
% unscented Kalman filter sigma points. % It is used in the function
% unscentedKalmanFilterCD.m, which implementes the filter detailed in
% https://doi.org/10.1109/TAC.2007.904453.
%
% Inputs:
%
%   t       The current time.
%
%   x       The current state estimate.
% 
%   u       The current input
% 
%   sqrtcP  The perturbation defining the sigma points.
%   
%   params  Parameters passed through to the dynamics.
%  
% Outputs:
%
%   dXdt    
%
%   dPdt
% 

% Compute the lower triangular Cholesky factor of P(t) satisfying Eq. (15).
Sx = chol(P)';

% Construct matrices of sigma points, their dynamics, and outputs.
[Xk,fX,hX] = sigmaPointsCT(t,X(:,1),u,sqrtc,Sx,f,h,params);

% Get the necessary dimensions.
nx = size(X,1);
ns = size(X,2);
nz = size(hX,1);
nv = size(Qc,1);

% Efficiently compute necessary covariances using Eqs. (36) and (37).
mx = zeros(nx,1);
mf = zeros(nx,1);
mh = zeros(nz,1);
for ii = 1:ns
    mx = mx + Wm(ii,1)*Xk(:,ii);
    mf = mf + Wm(ii,1)*fX(:,ii);
    mh = mh + Wm(ii,1)*hX(:,ii);
end
XWfXtr = zeros(nx,nx);
XWhXtr = zeros(nx,nz);
for ii = 1:ns
    XWfXtr = XWfXtr + Wc(ii,1)*(Xk(:,ii)-mx)*(fX(:,ii)-mf)';
    XWhXtr = XWhXtr + Wc(ii,1)*(Xk(:,ii)-mx)*(hX(:,ii)-mh)';
end

% compute the process noise input matrix
[~,~,D] = feval(f,t,X(:,1),u,zeros(nv,1),1,params);

% Compute the covariance dynamics using Eq. (34)
dPdt = XWfXtr + XWfXtr' + D*Qc*D';

% Compuite the square root prediction perturbations.
Sxinv = inv(Sx);
M = Sxinv*dPdt*Sxinv';
Phi = tril(M) - 0.5*diag(diag(M));
sqrtPerturb = sqrtc*[zeros(nx,1) Sx*Phi -Sx*Phi];

% Compute the dynamics of the sigma point predictions
dXdt = zeros(nx,ns);
for ii = 1:ns
    dXdt(:,ii) = fX*Wm + sqrtPerturb(:,ii);
end

end