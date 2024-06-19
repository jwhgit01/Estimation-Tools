function D = diffusionModel(t,x,u,params)
%diffusionModel
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This function implements diffusion matrix field, D, for the stochastic
% Lorenz system,
%
%                       dX = alpha*(Y - X)*dt                          (1a)
%                       dY = (X*(beta - Z) - Y)*dt                     (1b)
%                       dZ = (-gamma*Z + X*Y)*dt + sigma*dW            (1c)
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
%               alpha, beta, and gamma.
%  
% Outputs:
%
%   D           The value of the diffusion matrix field at time t.
%

D = [0;0;1];

end