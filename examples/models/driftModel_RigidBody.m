function [f,A] = driftModel_RigidBody(t,x,u,params)
%driftModel
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This function implements drift vector field, f, for a randomly forced
% rigid body in rotation using Euler angle kinematics
%
%  d(Theta) = LIB(Theta)*omega*dt                                      (1a)
%  d(omega) = inv(I)*(cross(I*omega,omega))*dt + RIB'(Theta)*Sigma*dW  (1b)
%
% Inputs:
%
%   t           The time at which f is evaluated
%
%   x           The 6 x 1 state vector at time t
%
%   u           The 0 x 1 control vector at time t (no control inputs)
%
%   params      A struct that contains values for the system parameters
%  
% Outputs:
%
%   f           The value of f at time t from Eq.(1)
%
%   A           The Jacobian of f with respect to x
%

% Get the known constants of the system
I = params.I;

% Parse the state vector
Theta = x(1:3,1);
omega = x(4:6,1);

% Rotation matrix
% R_IB = eul2rotm(flip(Theta.'),"ZYX");

% Attitude kinematics
phi = Theta(1);
theta = Theta(2);
L_IB = [1, sin(phi)*tan(theta), cos(phi)*tan(theta);...
        0, cos(phi),           -sin(phi)           ;...
        0, sin(phi)*sec(theta), cos(phi)*sec(theta)];

% Drift vector
f_Theta = L_IB*omega;
f_omega = I\cross(I*omega,omega);
f = [f_Theta;f_omega];

% Return if Jacobian is not needed
if nargout < 2
    return
end

% Jacobian of the drift vector field
q = omega(2);
r = omega(3);
A = zeros(6,6);
A(1,1) = (sin(theta)*(q*cos(phi) - r*sin(phi)))/cos(theta);
A(1,2) = (r*cos(phi) + q*sin(phi))/cos(theta)^2;
A(1:3,4:6) = L_IB;        
A(4:6,4:6) = I\(cpem(I*omega) - cpem(omega)*I);

end