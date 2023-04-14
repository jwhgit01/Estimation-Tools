function [Q,nu] = processNoisePSDTruthData(t,x,xcirc,u,nonlindyn,params)
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
%   nonlindyn   The function handle that computes either the
%               continuous-time dynamics of the system. The first line of
%               nonlindyn must be in the form
%                   [dxdt,A,D] = nonlindyn(t,x,u,vtil,dervflag,params)
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

% Loop through the samples and compute the modeled state derivative.
for k = 1:N
    tk = t(k,1);
    xk = x(k,:).';
    uk = u(k,:).';
    xdot(k,:) = nonlindyn(tk,xk,uk,[],0,params).';
end

% The difference between the modeled and nuumerical state derivative.
nu = xcirc - xdot;
dnu = nu - mean(nu,1);

% Bound the process noise power spectral desnity by its maximum value over
% the default frequency range returned by cpsd. First, compute the diagonal
% elements and initialize Q.
dt = t(2)-t(1);
pxx = pwelch(dnu,1/dt);
Q = diag(max(pxx));

% Loop through and compute the off-diagional elements of Q.
for ii = 1:nx
    for jj = (ii+1):nx
        pxy = cpsd(dnu(:,ii),dnu(:,jj),1/dt);
        pxymax = max(real(pxy));
        Q(ii,jj) = pxymax;
        Q(jj,ii) = pxymax;
    end
end

end