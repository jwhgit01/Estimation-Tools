function [J,Sinv,xhat,xbar,ytilde,Phi,Psi] = linearFilterErrorCost(sys,bias,xhat0,z,u,dt,K)

% Compute cost function for the filter error method, the covariance matrix of the
% measurement noise and its inverse, and the model outputs through state estimation.
% Simulation implies computation of state variables. 
% For linear systems it is done by state transition matrix
%
% Chapter 5: Filter Error Method
% "Flight Vehicle System Identification - A Time Domain Methodology"
% Author: Ravindra V. Jategaonkar
% Published by AIAA, Reston, VA 20191, USA
%
% Inputs:
%    Amat          system state matrix
%    Bmat          state input matrix
%    Cmat          state observation matrix
%    Dmat          input matrix
%    BX            lumped bias parameters of state equations
%    BY            lumped bias parameters of output equations
%    Ndata         number of data points
%    Ny            number of output variables
%    Nu            number of input variables
%    Nx            number of state variables
%    dt            sampling time
%    x0            initial conditions (=0 for linear systems with lumped bias parameters)
%    Uinp          measured inputs (Ndata,Nu) 
%    Z             measured outputs (Ndata,Ny)
%    param         parameter vector
%    Kgain         Kalman filter gain matrix
%
% Outputs:
%    currentcost   Value of current cost function (determinant of R)
%    R             covariance matrix
%    RI            inverse of covariance matrix
%    Y             computed (model) outputs (Ndata,Ny)
%    SXtilde       array of predicted state variables (Ndata,Nx)
%    SXhat         array of corrected state variables (Ndata, Nx)
%    SZmY          array of residuals (Ndata,Ny)
%    Phi           state transiton matrix (Nx,Nx)
%    Chi           integral of state transition matrix

% Get necessary state equation matrices and bias vectors
A = sys.A;
B = sys.B;
C = sys.C;
D = sys.D;
bx = bias.x;
by = bias.y;

% Get dimensions
N = size(z,1);
ny = size(z,2);
nx = size(xhat0,1);

% If K is empty, just propogate state vector.
if isempty(K)
    K = zeros(nx,ny);
end

% Approximate the state transition matrix and its integral using a Taylor series
Phi = eye(nx);
Psi = eye(nx)*dt;
for k = 1:12
    Phi = Phi + (A^k)*(dt^k)/(factorial(k));
    Psi = Psi + (A^k)*(dt^(k+1))/(factorial(k+1));
end   

% State estimation using a steady state Kalman filter
% ud = [u, repmat(bx.',N+1,1), repmat(by.',N+1,1)];
% Ad = Phi;
% Bd = [Psi*B, Psi, zeros(nx,ny)];
% Cd = C;
% Dd = [D, zeros(ny,nx), eye(ny)];
% [xhat,~,xbar,ybar] = linearStateObserverDT(z,ud,Ad,Bd,Cd,Dd,K,xhat0);
xhat = zeros(N,nx);
xbar = zeros(N,nx);
ybar = zeros(N,ny);
xbark = xhat0;
for k = 1:N
    uk = u(k,:).';
    ybark = C*xbark + D*uk + by;
    xbar(k,:) = xbark.';
    ybar(k,:) = ybark.';
    ytildek = z(k,:).' - ybark;
    xhatk = xbark + K*ytildek;
    xhat(k,:) = xhatk.';
    if k < N
        ubar = (uk + u(k+1,:).')/2;
        xbark = Phi*xhatk + Psi*B*ubar + Psi*bx;
    end
end

% The covariance of the ouput residuals is used to compute the cost, J
ytilde = z - ybar;
S = zeros(ny,ny);
for k = 1:N
    S = S + ytilde(k,:)'*ytilde(k,:);
end
S = diag(diag(S))/N;
cholS = chol(S);
Sinv = cholS\(cholS'\eye(ny));
%S = cov(ytilde);
J = det(S);

end