function [y,H] = measurementModel_GN(t,theta,x,params)
%measurementModel_GN
%
% Copyright (c) 2024 Jeremy W. Hopwood.  All rights reserved.
% 
% This function implements the nonlinear measurement model
%
%                   y = alpha + beta*exp(-gamma*x)                      (1)
%
% where theta = [alpha; beta; gamma]
% 
% Inputs:
%
%   t           Not used.
%
%   theta       The 3 x 1 parameter vector at time t.
%
%   x           The known value of x in (1).
%
%   params      Not used.
%  
%  Outputs:
%
%   y           The nz x 1 output at time t.
%
%   H           The nz x nx Jacobian matrix of h with respect to theta.
%

% Parameter vector, theta
alpha = theta(1,1);
beta = theta(2,1);
gamma = theta(3,1);

% Compute the output
y = alpha + beta*exp(-gamma*x);

% Return if derivative is not needed
if nargout < 2
    return
end

% Calculate the first derivative
H = [1, exp(-gamma*x), -beta*x*exp(-gamma*x)];

end