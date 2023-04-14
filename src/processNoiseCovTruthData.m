function [Q,nu] = processNoiseCovTruthData(t,x,xdot,u,nonlindyn,params)
%processNoiseCovTruthData
%
% Copyright (c) 2023 Jeremy W. Hopwood. All rights reserved.
%
% This function ...
%
% Inputs:
%
%   t           The N x 1 array of truth data sample times.
%
%   x           The N x nx array of truth data state vector values.
%
%   xdot        The N x nx array of (smoothed) numerical derivatives of the
%               time history of x.
%
%   u           The N x nu array of system inputs.
%
%   nonlindyn   The function handle that computes either the
%               continuous-time dynamics if t is given as a vector of
%               sample times or the discrete-time dynamics if t is empty.
%               The first line of nonlindyn must be in the form
%                   [dxdt,A,D] = nonlindyn(t,x,u,vtil,dervflag,params)
%               for a continuous-time system, or
%                   [xkp1,Fk,Gamk] = nonlindyn(k,xk,uk,vk,dervflag,params)
%               for a discrete-time system. If the system is given in
%               discrete-time, then the first sample must correspond to
%               sample number k = 0.
%
%   params      A struct of constants that get passed to the dynamics model
%               and measurement model functions.
%  
% Outputs:
%
%   Q           The approximated constant nx x nx process noise covariance.
%
%   nu          The N x nx time history of the difference between the
%               numerically-computed and modeled state time derivatives.
%

% Initialize necessary outputs.
N = size(x,1);
nx = sixe(x,2);
nu = zeros(N,nx);

% Check to see whether the system is discrete or continuous-time.
if isempty(t)
    DT = true;
else
    DT = false;
end

% Loop through the samples and compute the residual between the modeled and
% numerically-derived state dynamics.
for k = 1:N

    % Compute the modeled dynamics
    if DT
        tk = k-1;
    else
        tk = t(k,1);
    end
    xk = x(k,:).';
    uk = u(k,:).';
    xdotk_model = nonlindyn(tk,xk,uk,[],0,params).';

    % The difference between the model and the truth data.
    nu(k,:) = xdot(k,:) - xdotk_model;

end

% Assuming stationary noise, approximate the covariance.
Q = cov(nu);

end