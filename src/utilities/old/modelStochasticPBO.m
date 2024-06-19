function [dxhatdt,A,D] = modelStochasticPBO(t,xhat,u,y,~,params)
%modelStochasticPBO
%
% Copyright (c) 2023 Jeremy W. Hopwood. All rights reserved.
%
% This function is the "dynamic model" implemetation of the passivity-based
% observer presented in
%
%       Hopwood, To Appear.
%
% It is used in the function "stochasticPassivityBasedObserver.m", which
% runs this observer on measurement data. This function preserves the form
% of "nonlindyn_temp.m" for use in "c2dNonlinear.m", implementing
%
%           dxhat/dt = f(xhat,u) - L(y)*K(xhat,y,u)(x1hat-y)        (1)
%
% where x = [x1; x2], y = x1, L = [L1; L2(y)], and K has the form
%
%   K(xhat,y,u) = epsilon*eye(ny) ...
%               + Psi(x1hat-y,x2hat-(L2/L1)*(x1hat-y),y,u) ...
%               + {Lambda'*Lambda}(x1hat-y,x2hat-(L2/L1)*(x1hat-y),y,u) (2)
% 
% The gains epsilon, L1, L2, Psi, and Lambda are passed through to this
% function in the params struct as follows:
%
%   epsilon = params.epsilon    (constant positive scalar)
%   L1 = params.L1              (constant matrix)
%   L2 = params.L2              (function handle or constant matrix)
%   Psi = params.Psi            (function handle)
%   Lambda = parsms.Lambda      (function handle)
%
% The gains given as function handles must have the following form:
%
%       function     L2 = <function name>(y)
%       function    Psi = <function name>(x1tilde,eta2,x1,u)
%       function Lambda = <function name>(x1tilde,eta2,x1,u)
%
% where x1tilde = x1hat - x1. The nonlinear dynamics model, f(x,u), is also
% passed to this function through the params struct as follows:
%
%    nonlindyn = params.nonlindyn
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
L = params.L;
Q = params.Q;
Psi = params.Psi;
Lambda = params.Lambda;

% Get the nonlinear dynamics function handle.
nonlindyn = params.nonlindyn;

% Parse the state vector into its measured and un-measured parts.
x1hat = xhat(1:ny,1);
x2hat = xhat(ny+1:nx,1);

% Add x2hat to the params for the ability to gain schedule.
params.x2hat = x2hat;

% Check to see whether L is a function or constant matrix. If it is a
% function, evaluate it at the current measurement.
if isa(L,'function_handle')
    L = L(y,u,params);
end

% Evaluate the nonlinear dynamics at the state current estimate.
fhat = feval(nonlindyn,t,xhat,u,[],0,params);

% Compute the gain matrix, K.
x1tilde = x1hat - y;
eta2 = x2hat - L*x1tilde;
Psik = Psi(x1tilde,eta2,y,u,params);
Lambdak = Lambda(x1tilde,eta2,y,u,params);
K = epsilon*eye(ny) + Psik + Lambdak'*Lambdak;

% Compute the observer dynamics, dxhat/dt.
La = [eye(ny);L];
yd = x1tilde;
v = Q\(-K*yd);
dxhatdt = fhat + La*v;

end