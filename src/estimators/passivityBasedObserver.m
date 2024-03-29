function [xhat,phi1hist,phi2hist] = passivityBasedObserver(t,y,u,f,xhat0,epsilon,L1,L2,phi1,phi2,nRK,params)
%passivityBasedObserver 
%
% Copyright (c) 2023 Jeremy W. Hopwood. All rights reserved.
%
% This function performs passivity-based state observation on the system
%
%                       dx1/dt = f1(x1,x2,u)                        (1a)
%                       dx2/dt = f2(x1,x2,u)                        (1b)
%                            y = x1                                 (2)
% 
% as presented in
%
%       H. Shim, J.H. Seo, A.R. Teel, Nonlinear observer design via
%       passivation of error dynamics, Automatica, Volume 39, Issue 5,
%       2003, Pages 885-892
%       https://doi.org/10.1016/S0005-1098(03)00023-2
%
% The observer uses the function "modelPBO.m", which implmenents the
% state observer
%
%       dx1hat/dt = f1(x1hat,x2hat,u)-L1*k(xhat,y,u)(x1hat-y)       (3a)
%       dx2hat/dt = f2(x1hat,x2hat,u)-L2(y)*k(xhat,y,u)(x1hat-y)    (3b)
%
% where x = [x1; x2], L = [L1; L2(y)], and k has the form
%
%   k(xhat,y,u) = epsilon ...
%                 + phi1(x1hat-y,y,x2hat-(L2/L1)*(x1hat-y),u) ...
%                 + phi2^2(x1hat-y,y,x2hat-(L2/L1)*(x1hat-y),u)     (4)
%
% The gains given as function handles must have the following form:
%
%       function   L2 = <function name>(y)                          (5)
%       function phi1 = <function name>(x1tilde,x1,eta2,u)          (6a)
%       function phi2 = <function name>(x1tilde,x1,eta2,u)          (6b)
%
% where x1tilde = x1hat - x1 is denoted "e1" in Shim 2003. The nonlinear
% dynamics model, f(x,u), is also passed to this function through the
% params struct as a function handle.
%
% Inputs:
%
%   t       The N+1 x 1 sample time vector. The first element of t 
%           corresponds to the initial condition occuring at the same time
%           as the first measurement sample.
%
%   y       The N x ny time history of measurements. The first measurement
%           occurs at t=0.
%
%   u       The N x nu time history of system inputs. The first input
%           occurs at t=0. If not applicable
%           set to an empty array, [].
% 
%   f       The function handle that computes the continuous-time dynamics
%           of the system. The first line of f must be in the form
%               [f,A,D] = nonlindyn(t,x,u,vtil,dervflag,params)
%
%   xhat0   The nx x 1 initial state estimate occuring at t=0;.
%
%   epsilon A positive scalar added to the injection gain.
%
%   L1      The constant nx1 x nx1 gain matrix used in Eq. (3a).
%
%   L2      The nx2 x nx1 gain matrix used in Eq. (3b). It may be given as
%           a constant matrix or a function handle that is a function of y.
%
%   phi1    The handle of the function the computes the bounding function
%           phi1(x1hat-y,y,x2hat-(L2/L1)*(x1hat-y),u). It should have the
%           form given by Eq. (6a).
%
%   phi2    The handle of the function the computes the bounding function
%           phi2(x1hat-y,y,x2hat-(L2/L1)*(x1hat-y),u). It should have the
%           form given by Eq. (6b).
%
%   nRK     The number of intermediate Runge-Kutta integrations steps used
%           in the discretization of the nonlinear dynamics. May be given
%           as an empty array to use the default number.
%
%   params  A struct of constants that get passed to the dynamics model and
%           measurement model functions.
%  
% Outputs:
%
%   xhat        The (N+1) x nx array that contains the time history of the
%               state vector estimates. The N+1 element occurs one time
%               step after the last measurement/input at t(N+1).
%
%   phi1hist    The N x 1 array that contains the time history of the
%               positive bounding function phi1. The first element of
%               phi1hist is the gain used from t=0 to t=dt.
%
%   phi2hist    The N x 1 array that contains the time history of the
%               positive bounding function phi2. The first element of
%               phi2hist is the gain used from t=0 to t=dt.
%

% number of runge-kutta integration steps
if isempty(nRK)
    nRK = 10;
end

% Get the problem dimensions and initialize the output arrays.
N = size(y,1);
nx = size(xhat0,1);
ny = size(y,2);
xhat = zeros(N,nx);
xhat(1,:) = xhat0.';
phi1hist = zeros(N,1);
phi2hist = zeros(N,1);

% if no inputs, set to zero
if isempty(u)
    u = zeros(N,1);
end

% Check if L2 is a function.
if isa(L2,'function_handle')
    L2fun = true;
else
    L2fun = false;
end

% Store the necessary gains in the params struct for use in modelPBO.m
params.epsilon = epsilon;
params.L1 = L1;
params.L2 = L2;
params.phi1 = phi1;
params.phi2 = phi2;
params.nonlindyn = f;

% This loop integreates the continuous-time observer from one measurement
% to the next.
for k = 1:N-1
  
    % Obtain the current state estimate, inputs, and measurements
    disp(['k = ' num2str(k)])
    xhatk = xhat(k,:).';
    disp(xhatk')
    yk = y(k,:).';
    uk = u(k,:).';
    tk = t(k,1);
    tkp1 = t(k+1,1);

    % Compute the gains phi1 and phi2 for the current time step and store.
    % Add x2hat to the params for the ability to gain schedule.
    x1hatk = xhatk(1:ny,1);
    x2hatk = xhatk(ny+1:nx,1);
    %
%     figure(1)
%     plot(tk,x1hatk(1:3),'or')
%     plot(tk,x1hatk(4:6),'ob')
%     plot(tk,x1hatk(7:9),'og')
%     plot(tk,x1hatk(10:12),'oc')
%     figure(2)
%     plot(tk,x2hatk(1:3),'or')
%     plot(tk,x2hatk(4:6),'ob')
    %
    params.x2hat = x2hatk;
    if L2fun
        L2k = L2(yk,uk,params);
    else
        L2k = L2;
    end
    x1tildek = x1hatk - yk;
    eta2k = x2hatk - (L2k/L1)*x1tildek;
    phi1k = phi1(x1tildek,yk,eta2k,uk,params);
    phi2k = phi2(x1tildek,yk,eta2k,uk,params);
    phi1hist(k,1) = phi1k;
    phi2hist(k,1) = phi2k;
    disp(['phi1 = ' num2str(phi1k) ', phi2 = ' num2str(phi2k)])

    % Integrate the observer to the next meaurement and store the result.
    xhatkp1 = c2dNonlinear(xhatk,uk,yk,tk,tkp1,nRK,@modelPBO,0,params);
    xhat(k+1,:) = xhatkp1.';

end

end