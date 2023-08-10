function [f,A,D] = nonlindyn_CT(t,x,u,vtil,dervflag,params)
%nonlindyn_CT
%
% Copyright (c) 2023 Jeremy W. Hopwood. All rights reserved.
%
% This function implements a stochastic version of the Lorenz system,
%
%                       dX/dt = alpha*(Y - X)                       (1a)
%                       dY/dt = X*(beta - Z) - Y                    (1b)
%                       dZ/dt = -gamma*Z + X*Y + sigma*vtil         (1c)
%
% represented by the the continuous-time, nonlinear model
%
%                       dx/dt = f(t,x,vtil)                          (2)
%
% where vtil(t) is continuous-time white noise and x = [X; Y; Z].
%
% Inputs:
%
%   t           The time at which dx/dt is evaluated.
%
%   x           The 3 x 1 state vector at time t.
%
%   u           The 0 x 1 control vector at time t (no control inputs).
%
%   vtil        The scalar-valued process noise at time t.
%
%   dervflag    A flag that determines whether (dervflag = 1) or not
%               (dervflag = 0) the partial derivatives df/dx and df/dvtil
%               must be calculated. If dervflag = 0, then these outputs
%               will be empty arrays.
%
%   params      A struct that contains values for the system parameters,
%               alpha, beta, gamma, and sigma.
%  
% Outputs:
%
%   f           The time derivative of x at time t from Eq.(1).
%
%   A           The Jacobian of f with respect to x. It is evaluated
%               and output only if dervflag = 1. Otherwise, an empty array.
%
%   D           The Jacobian of f with respect to vtil. It is evaluated
%               and output only if dervflag = 1. Otherwise, an empty array.
%

% Get the known constants of the system.
alpha = params.alpha;
beta = params.beta;
gamma = params.gamma;
sigma = params.sigma;

% If vtil is empty, set it to zero.
if isempty(vtil)
    vtil = 0;
end

% Parse the state vector.
X = x(1,1);
Y = x(2,1);
Z = x(3,1);

% System dynamics.
Xdot = alpha*(Y - X);
Ydot = X*(beta - Z) - Y;
Zdot = -gamma*Z + X*Y + sigma*vtil;
f = [Xdot;Ydot;Zdot];

% Calculate the partial derivatives if they are needed.
if dervflag == 1
    A = [-alpha, alpha,      0;
         beta-Z,    -1,     -X;
              Y,     X, -gamma];
    D = [0;0;sigma];
else
    A = [];
    D = [];
end

end