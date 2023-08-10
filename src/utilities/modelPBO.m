function [dxhatdt,A,D] = modelPBO(t,xhat,u,y,~,params)
%modelPBO
%
% Copyright (c) 2023 Jeremy W. Hopwood. All rights reserved.
%
% This function is the "dynamic model" implemetation of the passivity-based
% observer presented in
%
%       H. Shim, J.H. Seo, A.R. Teel, Nonlinear observer design via
%       passivation of error dynamics, Automatica, Volume 39, Issue 5,
%       2003, Pages 885-892
%       https://doi.org/10.1016/S0005-1098(03)00023-2
%
% It is used in the function "passivityBasedObserver.m", which runs this
% observer on measurement data. This function preserves the form of
% "nonlindyn_temp.m" for use in "c2dNonlinear.m", implementing the observer
%
%           dxhat/dt = f(xhat,u) - L(y)*k(xhat,y,u)(x1hat-y)        (1)
%
% where x = [x1; x2], y = x1, L = [L1; L2(y)], and k has the form
%
%   k(xhat,y,u) = epsilon ...
%                 + phi1(x1hat-y,y,x2hat-(L2/L1)*(x1hat-y),u) ...
%                 + phi2^2(x1hat-y,y,x2hat-(L2/L1)*(x1hat-y),u)     (2)
% 
% The gains epsilon, L1, L2, phi, and phi2 are passed through to this
% function in the params struct as follows:
%
%   epsilon = params.epsilon    (constant positive scalar)
%   L1 = params.L1              (constant matrix)
%   L2 = params.L2              (function handle or constant matrix)
%   phi1 = params.phi1          (function handle)
%   phi2 = parsms.phi2          (function handle)
%
% The gains given as function handles must have the following form:
%
%       function   L2 = <function name>(y)
%       function phi1 = <function name>(x1tilde,x1,eta2,u)
%       function phi2 = <function name>(x1tilde,x1,eta2,u)
%
% where x1tilde = x1hat - x1 is denoted "e1" in Shim 2003. The nonlinear
% dynamics model, f(x,u), is also passed to this function through the
% params struct as a function handle as follows:
%
%   nonlindyn = params.nonlindyn
%
% The inputs and outputs of this function are as follows:
%
% Inputs:
%
%   t           The time at which dxhat/dt is evaluated.
%
%   xhat        The nx x 1 state vector estimate at time t.
%
%   u           The nu x 1 control vector at time t.
%
%   y           The ny x 1 measurement vector at time t.
%
%   dervflag    ~, not used.
%
%   params      A struct array that is used to pass through constant
%               parameters to the dynamics model as well as to store the
%               gains epsilon, L1, L2, phi1, and phi2.
%  
% Outputs:
%
%   f           The time derivative of xhat at time t from Eq.(1).
%
%   A           ~, not used.
%
%   D           ~, not used.
%

% Set un-used output.
A = [];
D = [];

% Get neccessary dimensions.
nx = size(xhat,1);
ny = size(y,1);

% Get the observer gains.
epsilon = params.epsilon;
L1 = params.L1;
L2 = params.L2;
phi1 = params.phi1;
phi2 = params.phi2;

% Get the nonlinear dynamics function handle.
nonlindyn = params.nonlindyn;

% Parse the state vector into its measured and un-measured parts.
x1hat = xhat(1:ny,1);
x2hat = xhat(ny+1:nx,1);

% Add x2hat to the params for the ability to gain schedule.
params.x2hat = x2hat;

% Check to see whether L2 is a function or constant matrix. If it is a
% function, evaluate it at the current measurement.
if isa(L2,'function_handle')
    L2 = L2(y,u,params);
end

% Evaluate the nonlinear dynamics at the state current estimate.
fhat = feval(nonlindyn,t,xhat,u,y,0,params);

% Compute the scalar injection gain, k.
x1tilde = x1hat - y;
eta2 = x2hat - (L2/L1)*x1tilde;
k = epsilon + phi1(x1tilde,y,eta2,u,params) + phi2(x1tilde,y,eta2,u,params)^2;

% Compute the observer dynamics, dxhat/dt.
L = [L1;L2];
yd = x1tilde;
v = -k*yd;
dxhatdt = fhat + L*v;

end