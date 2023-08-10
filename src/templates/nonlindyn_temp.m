function [f,A,D] = nonlindyn_temp(t,x,u,vtil,dervflag,params)
%nonlindyn_temp
%
% Copyright (c) 2023 Jeremy W. Hopwood. All rights reserved.
%
% This is a template function for a continuous-time, nonlinear model of a
% dynamical control system,
%
%                       dx/dt = f(t,x,u,vtil)                       (1)
%
% where vtil(t) is a continuous-time random process. Alternatively this
% function may define a discrete-time difference equation,
%
%                       x(k+1) = f(k,x(k),u(k),v(k))                (2)
%
% where v(k) is a discrete-time random process.
%
% Inputs:
%
%   t           The time at which dx/dt is evaluated. If the system is
%               discrete, t=k is the sample number, where the initial
%               condition occurs at k=0.
%
%   x           The nx x 1 state vector at time t (or sample k).
%
%   u           The nu x 1 control vector at time t (or sample k).
%
%   vtil        The nv x 1 process noise vector at time t (or sample k).
%
%   dervflag    A flag that determines whether (dervflag = 1) or not
%               (dervflag = 0) the partial derivatives df/dx and df/dvtil
%               must be calculated. If dervflag = 0, then these outputs
%               will be empty arrays.
%
%   params      Any data type that is used to pass through constant
%               parameters to the dynamics model.
%  
% Outputs:
%
%   f           The time derivative of x at time t from Eq.(1). If
%               discrete-time, f =: x(k+1) := fk is the value of the state
%               vector at the next sample.
%
%   A           The Jacobian of f with respect to x. It is evaluated
%               and output only if dervflag = 1. Otherwise, an empty array
%               is output.
%
%   D           The Jacobian of f with respect to vtil. It is evaluated
%               and output only if dervflag = 1. Otherwise, an empty array
%               is output.
%

% The necessary dimensions of the system.
nx = 2;
nu = 1;
nv = 1;

% Get the known constants of the system.
m = params.m;
k = params.k;
b = params.b;

% If vtil is empty, set it to zero.
if isempty(vtil)
    vtil = zeros(nv,1);
end

% If u is empty, set it to zero.
if isempty(u)
    u = zeros(nu,1);
end

% Evaluate the differential equations.
f = zeros(nx,1);
f(1,1) = x(2,1);
f(2,1) = -b/m*x(2,1) - k/m*x(1,1) + u + vtil;

% Calculate the partial derivative if they are needed.
if dervflag == 1
    A = [0,1;-k/m,-b/m];
    D = [0;1];
else
    A = [];
    D = [];
end

end