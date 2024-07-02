function [D,J] = diffusionModel_RigidBody(t,x,u,params)
%diffusionModel_RigidBody
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This function implements diffusion matrix field, D, for a randomly forced
% rigid body in rotation using Euler angle kinematics
%
%  d(Theta) = LIB(Theta)*omega*dt                                      (1a)
%  d(omega) = inv(I)*(cross(I*omega,omega))*dt + RIB'(Theta)*Sigma*dW  (1b)
%
% Inputs:
%
%   t           The time at which f is evaluated
%
%   x           The 3 x 1 state vector at time t
%
%   u           The 0 x 1 control vector at time t (no control inputs)
%
%   params      A struct that contains values for the system parameters,
%               alpha, beta, and gamma.
%  
% Outputs:
%
%   D           The value of the diffusion matrix field at time t.
%
%   J           The Jacobian of D with respect to x
%

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

% Diffusion matrix
D = [zeros(3); R_IB'];

% Return if Jacobian is not needed
if nargout < 2
    return
end

% Jacobian of the D matrix
phi = Theta(1);
theta = Theta(2);
psi = Theta(3);
J = zeros(6,3,6,'like',x);
J(5:6,1:3,1) = [sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta),   cos(phi)*sin(psi)*sin(theta) - cos(psi)*sin(phi),  cos(phi)*cos(theta);
                cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(theta), - cos(phi)*cos(psi) - sin(phi)*sin(psi)*sin(theta), -cos(theta)*sin(phi)];
J(4:6,1:3,2) = [        -cos(psi)*sin(theta),         -sin(psi)*sin(theta),          -cos(theta);
                cos(psi)*cos(theta)*sin(phi), cos(theta)*sin(phi)*sin(psi), -sin(phi)*sin(theta);
                cos(phi)*cos(psi)*cos(theta), cos(phi)*cos(theta)*sin(psi), -cos(phi)*sin(theta)];
J(4:6,1:2,3) = [                             -cos(theta)*sin(psi),                              cos(psi)*cos(theta);
                -cos(phi)*cos(psi) - sin(phi)*sin(psi)*sin(theta), cos(psi)*sin(phi)*sin(theta) - cos(phi)*sin(psi);
                 cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta), sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)];
