function [y,H] = measmodel_temp(t,x,u,dervflag,params)
%
%  Copyright (c) 2023 Jeremy W. Hopwood.  All rights reserved.
% 
%  This function is a template for specifying the measurement model,
%
%       Continuous-time:    y(t) = h(t,x,u)                         (1)
%         Discrete-time:    y(k) = h(k,x(k),u(k))                   (2)
%    Continous-discrete:   y(tk) = h(tk,x(tk),u(tk))                (3)
%
%  It is assumed for most filtering problems that measurement noise, w, is
%  additive such that
%
%                           z(k) = h(k,x(k),u(k)) + w(k)            (4)
%
%  and thus no information about w is required in this function. 
%  h(t,x,u), and its derivative with respect to x, H = dh/dx. This function
%  is for use in either static estimation problems such as nonlinear
%  least-squares or dyamic estimation problems such as extended Kalman
%  filtering.
%
%  Inputs:
%
%   t           The time at which h is evaluated. If the system is
%               discrete, t=k is the sample number, where the initial
%               condition occurs at k=0.
%
%   x           The nx x 1 state vector at time t (or sample k).
%
%   u           The nu x 1 control vector at time t (or sample k).
%
%   dervflag    A flag that determines whether (dervflag = 1) or not
%               (dervflag = 0) the partial derivative dh/dx must be
%               calculated. If dervflag = 0, then this output will be
%               an empty array.
%
%   params      Any data type that is used to pass through constant
%               parameters to the dynamics model.
%  
%  Outputs:
%
%   y           The nz x 1 output vector at time t or sample k.
%
%   H           The nz x nx Jacobian matrix of h with respect to x.
%               This output is needed  when performaing extended Kalman
%               filtering or Gauss-Newton estimaton, for example.
%

% Define the dimension of x and y.
ny = 1;
nx = 2;

% Set up output arrays as needed.
y = zeros(ny,1);
if dervflag == 0
    H = zeros(ny,nx);
else
    H = [];
end

% Compute the outputs.
y(1,1) = x(1);

% Return if neither first derivatives nor second derivatives need to be
% calculated.
if dervflag == 0
    return
end

% Calculate the first derivative.
H(1,1) = 1;
H(1,2) = 0;

end