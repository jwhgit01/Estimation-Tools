function [Q,nu] = processNoiseCovTruthData(t,x,u,nonlindyn,params)
%processNoiseCovTruthData
%
% Copyright (c) 2023 Jeremy W. Hopwood. All rights reserved.
%
% This function ...
%
% Inputs:
%
%   t           The N x 1 array of truth data sample times. If nonlindyn is
%               a discrete-time system, set as the empty array t = [].
%
%   x           The N x nx array of truth data state vector values.
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
%   Q           The approximated constant discrete-time nx x nx process
%               noise covariance.
%
%   nu          The N x nx time history of the difference between the
%               numerically-computed and modeled state time derivatives.
%

% Initialize necessary outputs.
N = size(x,1);
nx = size(x,2);
fk = zeros(N-1,nx);

% Check to see whether the system is discrete or continuous-time.
if isempty(t)
    DT = true;
else
    DT = false;
end

% Loop through the samples and compute the modeled state propogation.
for k = 0:N-2

    % Arrays are 1-indexed, but the first sample occurs at k = 0.
    kp1 = k+1;

    % Compute the modeled dynamics
    xk = x(kp1,:).';
    uk = u(kp1,:).';
    if DT
        fk(kp1,:) = nonlindyn(k,xk,uk,[],0,params).';
    else
        tk = t(kp1,1);
        tkp1 = t(kp1+1,1);
        fk(kp1,:) = c2dNonlinear(xk,uk,[],tk,tkp1,10,nonlindyn,0,params);
    end
    
end

% The difference between the modeled and actual state propogation.
xkp1 = x(2:N,:);
nu = xkp1 - fk;

% Assuming stationary noise, approximate the covariance.
Q = cov(nu);

end