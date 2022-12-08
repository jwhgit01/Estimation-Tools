function [Xk,fX,hX] = sigmaPointsCT(tk,xk,uk,sqrtc,Sx,f,h,params)
%sigmaPointsCT 
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function computes matrices of sigma points, their dynamics, and
% outputs given the perturbation sqrtcP = sqrt(c)*chol(P)'.
%
% Inputs:
%
%   t       The current time.
%
%   x       The current state estimate.
% 
%   u       The current input
% 
%   sqrtcP  The perturbation defining the sigma points.
%   
%   params  Parameters passed through to the dynamics.
%  
% Outputs:
%
%   Xk      The (nx)x(2*nx+1) array that contains the sigma points.
%
%   fk      The (nx)x(2*nx+1) array that contains the dynamics evaluated
%           at the matrix of sigma points.
%
%   hk      The (nz)x(2*nx+1) array that contains the output evaluated
%           at the matrix of sigma points.
% 

% Evaluate the dynamics and output at the given estimate, x.
fX0 = feval(f,tk,xk,uk,[],0,params);
hX0 = feval(h,tk,xk,uk,0,params);

% dimensions
nx = size(fX0,1);
nz = size(hX0,1);

% Construct matrices of sigma points, their dynamics, and outputs.
Xk = zeros(nx,2*nx+1);
Xk(:,1) = xk;
fX = zeros(nx,2*nx+1);
fX(:,1) = fX0;
hX = zeros(nz,2*nx+1);
hX(:,1) = hX0;
for ii = 2:(nx+1)
    Xk(:,ii) = xk + sqrtc*Sx(:,ii-1);
    fX(:,ii) = feval(f,tk,Xk(:,ii),uk,[],0,params);
    hX(:,ii) = feval(h,tk,Xk(:,ii),uk,0,params);
end
for ii = (nx+2):(2*nx+1)
    Xk(:,ii) = xk - sqrtc*Sx(:,ii-1-nx);
    fX(:,ii) = feval(f,tk,Xk(:,ii),uk,[],0,params);
    hX(:,ii) = feval(h,tk,Xk(:,ii),uk,0,params);
end

end