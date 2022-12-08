function [xbarkp1,Pbarkp1] = predictEKBF(xhatk,uk,Pk,Qc,tk,tkp1,nRK,fc,params)
%predictEKBF 
%
%  Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
%  This function ...
%
%  Inputs:
%
%    TODO
%  
%  Outputs:
%
%    TODO        
% 

% Prepare for the Runge-Kutta numerical integration by setting up 
% the initial conditions and the time step.
x = xhatk;
P = Pk;
t = tk;
delt = (tkp1-tk)/nRK;

% This loop does one 4th-order Runge-Kutta numerical integration step
% per iteration.  Integrate the state.  If partial derivatives are
% to be calculated, then the partial derivative matrices simultaneously
% with the state.
for jj = 1:nRK

    % Step a
    [f,A,D] = feval(fc,t,x,uk,[],1,params);
    Pdot = A*P + P*A' + D*Qc(t)*D';
    dxa = f*delt;
    dPa = Pdot*delt;

    % Step b
    [f,A,D] = feval(fc,t+0.5*delt,x+0.5*dxa,uk,[],1,params);
    Pdot = A*(P+0.5*dPa) + (P+0.5*dPa)*A' + D*Qc(t+0.5*delt)*D';
    dxb = f*delt;
    dPb = Pdot*delt;

    % Step c
    [f,A,D] = feval(fc,t+0.5*delt,x+0.5*dxb,uk,[],1,params);
    Pdot = A*(P+0.5*dPb) + (P+0.5*dPb)*A' + D*Qc(t+0.5*delt)*D';
    dxc = f*delt;
    dPc = Pdot*delt;

    % Step d
    [f,A,D] = feval(fc,t+delt,x+dxc,uk,[],1,params);
    Pdot = A*(P+dPc) + (P+dPc)*A' + D*Qc(t+delt)*D';
    dxd = f*delt;
    dPd = Pdot*delt;

    % 4th order Runge-Kutta integration result
    x = x + (dxa + 2*(dxb + dxc) + dxd)/6;
    P = P + (dPa + 2*(dPb + dPc) + dPd)/6;
    t = t + delt;

end

% Assign the results to the appropriate outputs.
xbarkp1 = x;
Pbarkp1 = P;

end