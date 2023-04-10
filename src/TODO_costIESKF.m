function [J,detadxi] = costIESKF(tk,tkp1,xk,xhatk,vk,vhatk,uk,ukp1,pk,fa,ha,nRK,params)
%costIESKF
%
%  Copyright (c) 2023 Jeremy W. Hopwood.  All rights reserved.
% 
%  This function ...
%   xa = [x(k); p]
% zeta = [xhat(k); vbar(k); z(k+1)]
%   xi = [x(k); v(k)];
%  eta = [x(k);v(k);h(k+1,f(k,x(k),u(k),v(k)),u(k+1))]
%
%  Inputs:
%
%   
%  Outputs:
%
%   
%

% Get dimensions.
nx = size(xk,1);
nv = size(vk,1);
np = size(pk,1);
nxi = nx + nv;
nzeta = nx + nv + nz;

% Check to see whether f is given as a difference or differential equation.
if isempty(tkp1)
    DT = true;
else
    DT = false;
end

% Check to see whether we need to compute derivatives
if nargout > 1
    ideriv = 1;
else
    ideriv = 0;
end

% Compute dynamics and output.
xak = [xk;pk];
if DT
    [xabarkp1,Fak,Gamk] = feval(fa,tk,xak,uk,vk,ideriv,params);
    [zbarkp1,Hkp1] = feval(ha,kp1,xakp1,ukp1,ideriv,params);
else
    [xabarkp1,Fak,Gamk] = c2dNonlinear(xak,uk,vk,tk,tkp1,nRK,f,ideriv,params);
    [zbarkp1,Hkp1] = feval(ha,tkp1,xakp1,ukp1,ideriv,params);
end
xbarkp1 = xabarkp1(1:nx,1);

% Compute the cost function
xaminusxahat = xak
zminuszbar = 
J = ????

% Compute derivative if necessary. Return empty array otherwise.
if nargout > 1
    detadxi = [eye(nxi);Hkp1*Fak,Hkp1*Gamk];
end

end