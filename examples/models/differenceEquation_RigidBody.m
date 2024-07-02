function [xkp1,F,G] = differenceEquation_RigidBody(k,xk,uk,wk,params)
%differenceEquation_RigidBody
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This function implements the discretized version of a randomly forced
% rigid body in rotation using Euler angle kinematics
%
%   d(Theta) = LIB(Theta)*omega*dt                                     (1a)
%   d(omega) = inv(I)*(cross(I*omega,omega))*dt + RIB(Theta)*Sigma*dW  (1b)
%
% Inputs:
%
%   k           The sample number at which f is evaluated
%
%   xk          The 6 x 1 state vector at sample k
%
%   uk          The 0 x 1 control vector at tsample k (no control inputs)
%
%   params      A struct that contains values for the system parameters
%  
% Outputs:
%
%   xkp1        The value of x at the next sample
%
%   A           The Jacobian of f with respect to x
%

% Fixed discretization time step
dt = params.dt;

% Drift vector
if nargout > 1
    [f,A] = driftModel_RigidBody([],xk,uk,params);
else
    f = driftModel_RigidBody([],xk,uk,params);
end

% Diffusion matrix
if ~isempty(wk) || nargout > 1
    G = diffusionModel_RigidBody([],xk,uk,params);
end

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
F = A*dt + eye(6);

end