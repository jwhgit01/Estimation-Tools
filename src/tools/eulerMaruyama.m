function [t,x] = eulerMaruyama(f,D,sig,x0,dt,tspan)
%eulerMaruyama Euler-Maruyama scheme for simulating SDEs
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This function implements the Euler-Maruyama scheme for simulating
% stochastic differential equations.
%
% Inputs:
%
%   in1     The first input...
%  
% Outputs:
%
%   out1    The first output...
%

% Set RNG to default seed for reproducability
rng("default");

% Dimensions
n = size(x0,1);
q = size(sig,2);

% Sampling times
t0 = tspan(1);
T = tspan(end);
N = ceil(T/dt);
t = (t0:dt:T).';
sqrtdt = sqrt(dt);

% Initialize result
x = zeros(N,n);
x(1,:) = x0.';

% Euler-Maruyama scheme
for k = 2:N
    xkm1 = x(k-1,:).';
    tkm1 = tEM(k-1);
    xk = xkm1 + f(tkm1,xkm1)*dt + D(tkm1,xkm1)*sig*sqrtdt*randn(q,1);
    x(k,:) = xk.';
end

% Re-sample if tspan is an array of times
if length(tspasn) > 2
    x = interp1(t,x,tspan(:),"previous");
    t = tspan(:);
end

end