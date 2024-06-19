function [xkp1,F,G] = differenceEquation(k,xk,uk,wk,params)
%differenceEquation
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This function implements the discretized version of the stochastic Lorenz
% system,
%
%                       dX = alpha*(Y - X)*dt                          (1a)
%                       dY = (X*(beta - Z) - Y)*dt                     (1b)
%                       dZ = (-gamma*Z + X*Y)*dt + sigma*dW            (1c)
%
% Inputs:
%
%   k           The sample number at which f is evaluated
%
%   xk          The 3 x 1 state vector at sample k
%
%   uk          The 0 x 1 control vector at tsample k (no control inputs)
%
%   params      A struct that contains values for the system parameters,
%               alpha, beta, and gamma
%  
% Outputs:
%
%   xkp1        The value of x at the next sample
%
%   A           The Jacobian of f with respect to x
%

% Fixed discretization time step
dt = params.dt;

% Get the known constants of the system
alpha = params.alpha;
beta = params.beta;
gamma = params.gamma;

% Parse the state vector
X = xk(1,1);
Y = xk(2,1);
Z = xk(3,1);

% Drift vector
fX = alpha*(Y - X);
fY = X*(beta - Z) - Y;
fZ = -gamma*Z + X*Y;
f = [fX;fY;fZ];

% Diffusion matrix
G = [0;0;1];

% Propogated state
if isempty(wk)
    xkp1 = xk + f*dt;
else
    xkp1 = xk + f*dt + G*wk;
end

% Return if Jacobian is not needed
if nargout < 2
    return
end

% Jacobian wrt the state
F = [-alpha, alpha,      0;
     beta-Z,    -1,     -X;
          Y,     X, -gamma]*dt + eye(3);

end