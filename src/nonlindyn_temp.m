function [f,A,D] = nonlindyn_temp(t,x,u,vtil,dervflag,params)
%nonlindyn_temp
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This is a template function for a continuous-time, nonlinear model of a
% dynamical control system,
%
%                       dx/dt = f(t,x,u,vtil)                       (1)
%
% where vtil(t) is a random process.
%
%   t               The time at which xdot is to be known, in seconds.
%
%   x               The nx1 cart state vector at time t.
%
%   u               The mx1 control vector at time t.
%
%   vtil            The qx1 process noise disturbance vector at time t.
%
%   dervflag        A flag that tells whether (dervflag = 1) or not
%                   (dervflag = 0) the partial derivatives df/dx and
%                   df/dvtil must be calculated. If dervflag = 0, then
%                   these outputs will be empty arrays.
%  
%  Outputs:
%
%   f               The time derivative of x at time t from Eq.(1).
%
%   A               The partial derivative of f with respect to x. This is
%                   a Jacobian matrix. It is evaluated and output only if
%                   dervflag = 1.  Otherwise, an empty array is output.
%
%   D               The partial derivative of f with respect to vtil. This
%                   is a Jacobian matrix. It is evaluated and output only
%                   if dervflag = 1.  Otherwise, an empty array is output.

% Set up the known constants of the system.
n = size(x,1);
m = 1;
k = 1;
b = 0.1;

% If vtil is empty, set it to zero.
if isempty(vtil)
    vtil = 0;
end

% If u is empty, set it to zero.
if isempty(u)
    u = 0;
end

% Evaluate the differential equations.
f = zeros(n,1);
f(1,1) = x(2);
f(2,1) = -b/m*x(2) - k/m*x(1) + u + vtil;

% Calculate the partial derivative if they are needed.
if dervflag == 1
    A = [0,1;-k/m,-b/m];
    D = [0;1];
else
    A = [];
    D = [];
end

end