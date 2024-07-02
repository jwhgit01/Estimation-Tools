function [f,A] = driftModel_Lorenz(t,x,u,params)
%driftModel
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This function implements drift vector field, f, for the stochastic Lorenz
% system,
%
%                   dX = alpha*(Y - X)*dt                              (1a)
%                   dY = (X*(beta - Z) - Y)*dt                         (1b)
%                   dZ = (-gamma*Z + X*Y)*dt + sigma*dW                (1c)
%
% Inputs:
%
%   t           The time at which f is evaluated
%
%   x           The 3 x 1 state vector at time t
%
%   u           The 0 x 1 control vector at time t (no control inputs)
%
%   params      A struct that contains values for the system parameters,
%               alpha, beta, and gamma
%  
% Outputs:
%
%   f           The value of f at time t from Eq.(1)
%
%   A           The Jacobian of f with respect to x
%

% Get the known constants of the system
alpha = params.alpha;
beta = params.beta;
gamma = params.gamma;

% Parse the state vector
X = x(1,1);
Y = x(2,1);
Z = x(3,1);

% Drift vector
fX = alpha*(Y - X);
fY = X*(beta - Z) - Y;
fZ = -gamma*Z + X*Y;
f = [fX;fY;fZ];

% Return if Jacobian is not needed
if nargout < 2
    return
end

% Jacobian of the drift vector field
A = [-alpha, alpha,      0;
     beta-Z,    -1,     -X;
          Y,     X, -gamma];

end