function [y,H] = measurementModel(t,x,u,params)
%measurementModel
%
% Copyright (c) 2024 Jeremy W. Hopwood.  All rights reserved.
% 
% This function is implements the measurement model for a stochastic
% version of the Lorenz system,
%
%                   dX = alpha*(Y - X)*dt                              (1a)
%                   dY = (X*(beta - Z) - Y)*dt                         (1b)
%                   dZ = (-gamma*Z + X*Y)*dt + sigma*dW                (1c)
%
% The output of the system is
%
%                               y = h(x)                                (3)
%
% These outputs are corrupted by measurement noise and are represented in
% continous-time by
%
%                       z(t) = h(t,x(t),u(t)) + v(t)                   (4a)
%
% or discrete-time by
%
%                       z(k) = h(k,x(t(k)),u(t(k))) + v(k)             (4b)
%
% where for each t, v is independently sampled from a Gaussian distribution
% with zero mean and covariance, R. Here, h(x) is 
%
%                           h(x) = [X; Y];                              (5)
%
% Inputs:
%
%   t           The time at which h is evaluated. Since h does not depend
%               on t, it is also a valid discrete-time measurement model
%               implementing (4b)
%
%   x           The 3 x 1 state vector at time t.
%
%   u           The 0 x 1 control vector at time t (no control inputs).
%
%   params      A struct that contains values for the system parameters,
%               alpha, beta, and gamma.
%  
%  Outputs:
%
%   y           The 2 x 1 output vector at time t.
%
%   H           The nz x nx Jacobian matrix of h with respect to x.
%               This output is needed  when performaing extended Kalman
%               filtering or Gauss-Newton estimaton, for example.
%

% Define the dimension of x and y.
nx = 3;
ny = 2;

% Compute the output.
y = x(1:ny,:);

% Return if derivative is not needed.
if nargout < 2
    return
end

% Calculate the first derivative.
H = zeros(ny,nx);
H(1:ny,1:ny) = eye(ny);

end