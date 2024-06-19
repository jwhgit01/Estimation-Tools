function [K,Pxx] = linearFilterErrorKalmanGain(sys,Sinv,dt)

% Solve steady state Riccati equation for P:
% AP + PA' - PC'Rinv CP/dt + FF' = 0 using Potter's method;
% and compute Kalman gain matrix
%
% Chapter 5: Filter Error Method
% "Flight Vehicle System Identification - A Time Domain Methodology"
% Author: Ravindra V. Jategaonkar
% Published by AIAA, Reston, VA 20191, USA
%
% Filter error method for linear systems (ml_fem_linear)
%
% Inputs:
%    Amat          system state matrix
%    Cmat          state observation matrix
%    Fmat          process noise distribution matrix
%    Nx            number of state variables
%    Ny            number of output variables
%    Nu            number of input variables
%    Nparam        total number of parameters to be estimated 
%    dt            sampling time
%    param         parameter vector
%    xt            state vector
%    Uinp          measured inputs (Ndata,Nu) 
%    RI            inverse of covariance matrix
%
% Outputs:
%    Kgain         Kalman filter gain matrix
%    Cmat          linearized observation matrix

% Get necessary state equation matrices
A = sys.A;
C = sys.C;
F = sys.F;

% Dimensions
nx = size(A,1);

% Form the Hamiltonian matrix
FFT  = F*F';
CSIC = (C'*Sinv)*C/dt;
H  = [A,-FFT;-CSIC,-A'];                          % Eq. (5.15)

% Compute eigenvalues and eigenvectors of the Hamiltonian matrix
[eigVec, eigVal] = eig(H);

% Sort eigVal in ascending order; get indices of eigenvalues with +ve real parts
[~,Is] = sort( diag( real(eigVal) ) );

% Eigenvectors corresponding to eigenvalues with positive real part
eigVecN = zeros(size(eigVec));
for ii = 1:nx
    eigVecN(:,ii) = eigVec(:,Is(nx+ii)); 
end

% Partitioning of the matrix of eigenvectors, with eigenvectors corresponding
% to eigenvalues with positive real parts in the left partition.
X11 = eigVecN(1:nx,1:nx);
X21 = eigVecN(nx+1:nx+nx,1:nx);

% Solution to Riccati equation
Pxx = real(-X11/X21); % Eq. (J.5.17)

% Compute Kalman gain matrix 
K = Pxx*C'*Sinv; % Eq. (J.5.8)

end
