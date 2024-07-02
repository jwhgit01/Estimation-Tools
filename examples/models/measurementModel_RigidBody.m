function [y,H] = measurementModel_RigidBody(t,x,u,params)
%measurementModel_RigidBody
%
% Copyright (c) 2024 Jeremy W. Hopwood.  All rights reserved.
% 
% This function is implements the measurement model for a randomly forced
% rigid body in rotation using Euler angle kinematics
%
%  d(Theta) = LIB(Theta)*omega*dt                                      (1a)
%  d(omega) = inv(I)*(cross(I*omega,omega))*dt + RIB'(Theta)*Sigma*dW  (1b)
%
% The output of the system is given by
%
%                            y1 = R_IB'(Theta)*e1                      (2a)
%                            y2 = R_IB'(Theta)*e3                      (2b)
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
% with zero mean and covariance, R.
%
% Inputs:
%
%   t           The time at which h is evaluated.
%
%   x           The 6 x 1 state vector at time t.
%
%   u           The 0 x 1 control vector at time t (no control inputs).
%
%   params      A struct that contains values for the system parameters.
%  
%  Outputs:
%
%   y           The 6 x 1 output vector at time t.
%
%   H           The nz x nx Jacobian matrix of h with respect to x.
%               This output is needed  when performaing extended Kalman
%               filtering or Gauss-Newton estimaton, for example.
%

% Define the dimension of x and y.
nx = 6;
ny = 6;

% Parse the state vector
Theta = x(1:3,1);

% Rotation matrix
phi = Theta(1);
theta = Theta(2);
psi = Theta(3);
R1 = [1,0,0;0,cos(phi),-sin(phi);0,sin(phi),cos(phi)];
R2 = [cos(theta),0,sin(theta);0,1,0;-sin(theta),0,cos(theta)];
R3 = [cos(psi),-sin(psi),0;sin(psi),cos(psi),0;0,0,1];
R_IB = R3*R2*R1;
% R_IB = eul2rotm(flip(Theta.'),"ZYX");

% Output
y1 = R_IB'*[1;0;0];
y2 = R_IB'*[0;0;1];
y = [y1; y2];

% Return if derivative is not needed.
if nargout < 2
    return
end

% Calculate the first derivative.
H = zeros(ny,nx);
H(1,2) = -cos(psi)*sin(theta);
H(1,3) = -cos(theta)*sin(psi);
H(2,1) = sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta);
H(2,2) = cos(psi)*cos(theta)*sin(phi);
H(2,3) = -cos(phi)*cos(psi) - sin(phi)*sin(psi)*sin(theta);
H(3,1) = cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(theta);
H(3,2) = cos(phi)*cos(psi)*cos(theta);
H(3,3) = cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta);
H(4,2) = -cos(theta);
H(5,1) = cos(phi)*cos(theta);
H(5,2) = -sin(phi)*sin(theta);
H(6,1) = -cos(theta)*sin(phi);
H(6,2) = -cos(phi)*sin(theta);
 


end