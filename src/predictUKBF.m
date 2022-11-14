function [xbarkp1,Pbarkp1,Xbarkp1,Ybarkp1] = predictUKBF(xbark,Pk,uk,Qc,tk,tkp1,nRK,f,h,sqrtc,Wm,Wc,params)
%predictUKBF 
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function ...
%
% Inputs:
%
%   Xk,Pk,uk,Qc,tk,tkp1,nRK,f,h,sqrtc,Wm,Wc,params
%  
% Outputs:
%
%   Xbarkp1
%
%   Pbarkp1
%
%   Xbarkp1
%
%   Ybarkp1
% 

% Prepare for the Runge-Kutta numerical integration by setting up 
% the initial conditions and the time step.
x = xbark;
P = Pk;
t = tk;
delt = (tkp1-tk)/nRK;

% This loop does one 4th-order Runge-Kutta numerical integration step
% per iteration.  Integrate the state.  If partial derivatives are
% to be calculated, then the partial derivative matrices simultaneously
% with the state.
for jj = 1:nRK

    % Step a
    [dxbardt,dPdt] = sigmaPointDynamicsCT(t,x,P,uk,f,h,Qc(t),sqrtc,Wm,Wc,params);
    dxbara = dxbardt*delt;
    dPa = dPdt*delt;

    % Step b
    [dxbardt,dPdt] = sigmaPointDynamicsCT(t+0.5*delt,x+0.5*dxbara,P+0.5*dPa,uk,f,h,Qc(t+0.5*delt),sqrtc,Wm,Wc,params);
    dxbarb = dxbardt*delt;
    dPb = dPdt*delt;

    % Step c
    [dxbardt,dPdt] = sigmaPointDynamicsCT(t+0.5*delt,x+0.5*dxbarb,P+0.5*dPb,uk,f,h,Qc(t+0.5*delt),sqrtc,Wm,Wc,params);
    dxbarc = dxbardt*delt;
    dPc = dPdt*delt;

    % Step d
    [dxbardt,dPdt] = sigmaPointDynamicsCT(t+delt,x+dxbarc,P+dPc,uk,f,h,Qc(t+delt),sqrtc,Wm,Wc,params);
    dxbard = dxbardt*delt;
    dPd = dPdt*delt;

    % 4th order Runge-Kutta integration result
    x = x + (dxbara + 2*(dxbarb + dxbarc) + dxbard)/6;
    P = P + (dPa + 2*(dPb + dPc) + dPd)/6;
    t = t + delt;

end

% Assign the results to the appropriate outputs.
xbarkp1 = x;
Pbarkp1 = P;

% Compute lower triangular Cholesky factor of Pk satisfying Eq. (15).
Sbarkp1 = chol(Pbarkp1)';

% Construct matrices of sigma point outputs.
[Xbarkp1,~,Ybarkp1] = sigmaPointsCT(t,xbarkp1,uk,sqrtc,Sbarkp1,f,h,params);


end