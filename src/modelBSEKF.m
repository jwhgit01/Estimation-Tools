function [eta,detadxi] = modelBSEKF(tk,tkp1,xk,vk,uk,ukp1,ideriv,params,f,h,nRK)
%modelBSEKF
%
%  Copyright (c) 2022 Jeremy W. Hopwood.  All rights reserved.
% 
%  This function ...
%
% zeta = [xhat(k); vbar(k); z(k+1)]
%   xi = [x(k); v(k)];
%  eta = [x(k);v(k);h(k+1,f(k,x(k),u(k),v(k)),u(k+1))]
%
%  Inputs:
%
%   x           The vector of initial conditions at time 0.
%
%   tj          The time in seconds of the measurement.
%
%   i1stdrv     A character array/string or boolean that that defines
%               whether or not and how Hj is to be computed. It follows the
%               following logic: If i1stdrv=false or i1stdrv=0, neither the
%               first nor second partial derivatves of hj are computed. If
%               i1stdrv=true or i1stdrv=1, the default method is 'Analytic'
%               in which this function defines the analytic formula for Hj.
%               Alternatively, i1stdrv may be 'Analytic', 'CSDA', 'CFDM',
%               or 'FiniteDifference' to explicitly specify the computation
%               method.
%
%   i2nddrv     A character array/string or boolean that that defines
%               whether or not and how d2hjdx2 is to be computed. It
%               follows the same logic as i1stdrv.
%  
%  Outputs:
%
%   hj          The nz x 1 output vector.
%
%   Hj          = dhj/dx. Hj is a pxn matrix. This output will be needed to
%               do Newton's method or to do the Gauss-Newton method.
%
%   d2hjdx2     = d2hj/dx2. d2hjdx2 is a pxnxn array. This output will be
%               needed to do Newton's method.
%

% Check to see whether f is given as a difference or differential equation.
if isempty(tkp1)
    DT = true;
    k = tk;
    kp1 = k+1;
else
    DT = false;
end

% Compute dynamics and output.
if DT
    [fk,Fk,Gamk] = feval(f,k,xk,uk,vk,ideriv,params);
    [hkp1,Hkp1] = feval(h,kp1,fk,ukp1,ideriv,params);
else
    [fk,Fk,Gamk] = c2dNonlinear(xk,uk,vk,tk,tkp1,nRK,f,ideriv,params);
    [hkp1,Hkp1] = feval(h,tkp1,fk,ukp1,ideriv,params);
end

% Get dimensions.
nx = size(xk,1);
nv = size(vk,1);
nz = size(hkp1,1);
nxi = nx + nv;
nzeta = nx + nv + nz;

% Compute the eta(xi) outputs.
eta(1:nx,1) = xk;
eta(nx+1:nxi,1) = vk;
eta(nxi+1:nzeta,1) = hkp1;

% Compute derivative if necessary. Return empty array otherwise.
if ideriv == 0
    detadxi = [];
else
    detadxi = [eye(nxi);Hkp1*Fk,Hkp1*Gamk];
end

end