function [y,H] = measmodel_CT(t,x,u,dervflag,params)
%measmodel_CT
%
% Copyright (c) 2023 Jeremy W. Hopwood.  All rights reserved.
% 
% This function is implements the measurement model for a stochastic
% version of the Lorenz system,
%
%                       dX/dt = alpha*(Y - X)                       (1a)
%                       dY/dt = X*(beta - Z) - Y                    (1b)
%                       dZ/dt = -gamma*Z + X*Y + sigma*vtil         (1c)
%
% represented by the the continuous-time, nonlinear model
%
%                       dx/dt = f(t,x,vtil)                          (2)
%
% where vtil(t) is continuous-time white noise and x = [X; Y; Z]. The
% output of the system is
%
%                           y = h(x)                                 (3)
%
% These outputs are corrupted by measurement noise and are represented by
%
%                        z(t) = h(x(t)) + w(t)                       (4)
%
% where for each t, w is independently sampled from a Gaussian distribution
% with zero mean and covariance, R. Here, h(x) is 
%
%                        h(x) = [X; Y];                              (5)
%
% Inputs:
%
%   t           The time at which h is evaluated.
%
%   x           The 3 x 1 state vector at time t.
%
%   u           The 0 x 1 control vector at time t (no control inputs).
%
%   dervflag    A flag that determines whether (dervflag = 1) or not
%               (dervflag = 0) the partial derivative dh/dx must be
%               calculated. If dervflag = 0, then this output will be
%               an empty array.
%
%   params      A struct that contains values for the system parameters,
%               alpha, beta, gamma, and sigma.
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

% Set up output arrays as needed.
if dervflag == 0
    H = [];
else
    H = zeros(ny,nx);
end

% Compute the output.
y = x(1:ny,:);

% Return if neither first derivatives nor second derivatives need to be
% calculated.
if dervflag == 0
    return
end

% Calculate the first derivative.
H(1:ny,1:ny) = eye(ny);

end