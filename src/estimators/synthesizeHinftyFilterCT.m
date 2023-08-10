function [K,gam,X,Y] = synthesizeHinftyFilterCT(A,G,C,D,L,gamma,idisp)
%synthesizeHinftyFilterCT
%
%  Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
%  This function synthesizes the continuious-time H_\infty filter
%  for the linear time-invariant system
%
%           dx/dt = A x(t) + B u(t) + G w(t)
%            y(t) = C x(t)          + D w(t)
%            z(t) = L x(t)
%
% such that the H_\infty norm of the transfer function
%
%                   z(s) - zhat(s)
%           H(s) = ----------------
%                        w(s)
%
% is less than or equal to gamma, where w is in the L2 vector space. If
% gamma is given, this function sysnthesizes the filter for the given fixed
% gamma. If it is not given, this function returns the filter that 
% minimizes gamma.
%
%  Inputs:
%
%    
%  
%  Outputs:
%
%    
% 

% Check to see if gamma is given
if nargin < 7
    gamma = [];
end

% dimensions 
nx = size(A,1);
nw = size(G,2);
ny = size(C,1);

% Set up CVX
% cvx_solver SDPT3
cvx_precision best
epsilon = 100*eps;

% solve the LMI as a SDP using CVX (MOSEK)
if idisp == 1
    cvx_begin sdp
else
    cvx_begin sdp quiet
end


    % create cvx variables
    variable X(nx,nx) semidefinite
    variable Y(nx,ny)

    % initialize delta = 1/gamma^2
    if isempty(gamma)
        variable delta nonnegative
        maximize delta
    else
        delta = 1/(gamma^2);
    end
    
    % Constraint equations
    subject to
        [A'*X+X*A-Y*C-C'*Y'+delta*(L'*L),X*G-Y*D;G'*X'-D'*Y',-eye(nw)] <= -epsilon*eye(nx+nw);
        X >= epsilon*eye(nx);

cvx_end

% return the results
gam = 1/sqrt(delta);
K = X\Y;
