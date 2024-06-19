function [dxsdt,dPsdt] = unscentedRTSDynamicsCT(t,xs,Ps,xhat,P,u,f,h,Qc,sqrtc,Wm,Wc,params)
%unscentedRTSDynamicsCT
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function computes 
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
%   dxsdt    
%
%   dPdt
% 

% Get the necessary dimensions.
nx = size(xhat,1);
ns = 2*nx+1;

% Compute the lower triangular Cholesky factor of P(t) satisfying Eq. (15).
Sx = chol(P)';

% Construct matrices of sigma points, their dynamics, and outputs.
[Xk,fX,~] = sigmaPointsCT(t,xhat,u,sqrtc,Sx,f,h,params);

% Efficiently compute necessary covariances.
mx = zeros(nx,1);
mf = zeros(nx,1);
for ii = 1:ns
    mx = mx + Wm(ii,1)*Xk(:,ii);
    mf = mf + Wm(ii,1)*fX(:,ii);
end
XWfXtr = zeros(nx,nx);
for ii = 1:ns
    XWfXtr = XWfXtr + Wc(ii,1)*(Xk(:,ii)-mx)*(fX(:,ii)-mf)';
end

% compute the process noise input matrix
[~,~,D] = feval(f,t,xhat,u,[],1,params);

% Compute the smoothing gain.
G = (XWfXtr' + D*Qc*D')/P;

% Compute the smoothed estimate dynamics.
dxsdt = mf + G*(xs-xhat);

% Compute the covariance dynamics.
dPsdt = G*Ps + Ps*G' - D*Qc*D';


end