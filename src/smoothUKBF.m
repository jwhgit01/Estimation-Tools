function [xskm1,Pskm1] = smoothUKBF(xsk,Psk,xhatk,Pk,u,Qc,tk,tkm1,nRK,f,h,sqrtc,Wm,Wc,params)
%smoothUKBF 
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
xs = xsk;
Ps = Psk;
t = tk;
delt = (tkm1-tk)/nRK;

% This loop does one 4th-order Runge-Kutta numerical integration step
% per iteration.  Integrate the state.  If partial derivatives are
% to be calculated, then the partial derivative matrices simultaneously
% with the state.
for jj = 1:nRK

    % Step a
    [dxsdt,dPsdt] = unscentedRTSDynamicsCT(t,xs,Ps,xhatk,Pk,u,f,h,Qc(t),sqrtc,Wm,Wc,params);
    dxsa = dxsdt*delt;
    dPsa = dPsdt*delt;

    % Step b
    [dxsdt,dPsdt] = unscentedRTSDynamicsCT(t+0.5*delt,xs+0.5*dxsa,Ps+0.5*dPsa,xhatk,Pk,u,f,h,Qc(t+0.5*delt),sqrtc,Wm,Wc,params);
    dxsb = dxsdt*delt;
    dPsb = dPsdt*delt;

    % Step c
    [dxsdt,dPsdt] = unscentedRTSDynamicsCT(t+0.5*delt,xs+0.5*dxsb,Ps+0.5*dPsb,xhatk,Pk,u,f,h,Qc(t+0.5*delt),sqrtc,Wm,Wc,params);
    dxsc = dxsdt*delt;
    dPsc = dPsdt*delt;

    % Step d
    [dxsdt,dPsdt] = unscentedRTSDynamicsCT(t+0.5*delt,xs+dxsc,Ps+dPsc,xhatk,Pk,u,f,h,Qc(t+delt),sqrtc,Wm,Wc,params);
    dxsd = dxsdt*delt;
    dPsd = dPsdt*delt;

    % 4th order Runge-Kutta integration result
    xs = xs + (dxsa + 2*(dxsb + dxsc) + dxsd)/6;
    Ps = Ps + (dPsa + 2*(dPsb + dPsc) + dPsd)/6;
    t = t + delt;

end

% Assign the results to the appropriate outputs.
xskm1 = xs;
Pskm1 = Ps;


end