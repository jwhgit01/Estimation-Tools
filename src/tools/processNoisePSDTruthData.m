function [Q,nu] = processNoisePSDTruthData(t,x,xcirc,u,f,params)
%processNoisePSDTruthData
%
% Copyright (c) 2023 Jeremy W. Hopwood. All rights reserved.
%
% This function ...
%
% Inputs:
%
%   t           The N x 1 array of truth data uniform sample times.
%
%   x           The N x nx array of truth data state vector values.
%
%   xcirc       The N x nx array of smoothed derivatives of the state
%               vector.
%
%   u           The N x nu array of system inputs.
%
%   f          The function handle that computes either the
%               continuous-time dynamics of the system. The first line of
%               nonlindyn must be in the form
%                   dxdt = drift(t,x,u,params)
%               If the system is linear with
%                   dxdt = A*x + B*u + w;
%               nonlindyn may be given as the cell array {A,B}.
%
%   params      A struct of constants that get passed to the dynamics model
%               and measurement model functions.
%  
% Outputs:
%
%   Q           The approximated constant continuous-time nx x nx process
%               noise power spectral density martrix.
%
%   nu          The N x nx time history of the difference between the
%               numerically-computed and modeled state time derivatives.
%

% Initialize necessary outputs.
N = size(x,1);
nx = size(x,2);
xdot = zeros(N,nx);

% Check to see if the system is linear
if iscell(f)
    LTI = true;
    A = f{1};
    B = f{2};
else
    LTI = false;
end

% Loop through the samples and compute the modeled state derivative.
for k = 1:N
    tk = t(k,1);
    xk = x(k,:).';
    uk = u(k,:).';
    if LTI
        xdot(k,:) = (A*xk + B*uk).';
    else
        xdot(k,:) = f(tk,xk,uk,params).';
    end
end

% The difference between the modeled and nuumerical state derivative.
nu = xcirc - xdot;
dnu = nu - mean(nu,1);

% Bound the process noise power spectral desnity by its maximum value over
% the default frequency range returned by cpsd. First, compute the diagonal
% elements and initialize Q.
dt = t(2)-t(1);
pxx = pwelch(dnu,[],[],[],1/dt);
Q = diag(max(pxx));

% Loop through and compute the off-diagional elements of Q.
for ii = 1:nx
    for jj = (ii+1):nx
        pxy = cpsd(dnu(:,ii),dnu(:,jj),[],[],[],1/dt);
        pxymax = max(real(pxy));
        Q(ii,jj) = pxymax;
        Q(jj,ii) = pxymax;
    end
end

end