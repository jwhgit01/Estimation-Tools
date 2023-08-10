function [Xbarkp1,Pbarkp1,Ybarkp1] = predictSqrtUKBF(Xk,Pk,uk,Qc,tk,tkp1,nRK,f,h,sqrtc,Wm,Wc,params)
%predictSqrtUKBF 
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
%   Ybarkp1
% 

% Prepare for the Runge-Kutta numerical integration by setting up 
% the initial conditions and the time step.
X = Xk;
P = Pk;
t = tk;
delt = (tkp1-tk)/nRK;

% This loop does one 4th-order Runge-Kutta numerical integration step
% per iteration.  Integrate the state.  If partial derivatives are
% to be calculated, then the partial derivative matrices simultaneously
% with the state.
for jj = 1:nRK

    % Step a
    [dXdt,dPdt] = sqrtSigmaPointDynamicsCT(t,X,P,uk,f,h,Qc,sqrtc,Wm,Wc,params);
    dXa = dXdt*delt;
    dPa = dPdt*delt;

    % Step b
    [dXdt,dPdt] = sqrtSigmaPointDynamicsCT(t+0.5*delt,X+0.5*dXa,P+0.5*dPa,uk,f,h,Qc,sqrtc,Wm,Wc,params);
    dXb = dXdt*delt;
    dPb = dPdt*delt;

    % Step c
    [dXdt,dPdt] = sqrtSigmaPointDynamicsCT(t+0.5*delt,X+0.5*dXb,P+0.5*dPb,uk,f,h,Qc,sqrtc,Wm,Wc,params);
    dXc = dXdt*delt;
    dPc = dPdt*delt;

    % Step d
    [dXdt,dPdt] = sqrtSigmaPointDynamicsCT(t+delt,X+dXc,P+dPc,uk,f,h,Qc,sqrtc,Wm,Wc,params);
    dXd = dXdt*delt;
    dPd = dPdt*delt;

    % 4th order Runge-Kutta integration result
    X = X + (dXa + 2*(dXb + dXc) + dXd)/6;
    P = P + (dPa + 2*(dPb + dPc) + dPd)/6;
    t = t + delt;

end

% Assign the results to the appropriate outputs.
Xbarkp1 = X;
Pbarkp1 = P;

% Evaluate the dynamics and output for the propogated dynamics.
fXbarkp10 = feval(f,tkp1,Xbarkp1(:,1),uk,[],0,params);
hXbarkp10 = feval(h,tkp1,Xbarkp1(:,1),0,0);

% dimensions
nx = size(fXbarkp10,1);
nz = size(hXbarkp10,1);

% Construct the output of the propogated dynamics.
hXbarkp1 = zeros(nz,2*nx+1);
hXbarkp1(:,1) = hXbarkp10;
for ii = 2:(nx+1)
    hXbarkp1(:,ii) = feval(h,tkp1,Xbarkp1(:,ii),0,0);
end
for ii = (nx+2):(2*nx+1)
    hXbarkp1(:,ii) = feval(h,tkp1,Xbarkp1(:,ii),0,0);
end

% Assign the results to the appropriate outputs.
Ybarkp1 = hXbarkp1;

end