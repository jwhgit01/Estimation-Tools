function dPdth = gradP(sys,jac,dt,K,Pxx)                 
% Compute gradients of covariance matrix P:
% This requires solution of the Lyapunov equations, once for each parameter to be estimated
%
% Chapter 5: Filter Error Method
% "Flight Vehicle System Identification - A Time Domain Methodology"
% Author: Ravindra V. Jategaonkar
% Published by AIAA, Reston, VA 20191, USA
%

% Get necessary state equation matrices
A = sys.A;
C = sys.C;
F = sys.F;

% Get necessary state equation Jacobians 
dAdth = jac.A;
dCdth = jac.C;
dFdth = jac.F;

% Dimensions
nx = size(A,1);
nth = size(dAdth,3);

% Initialize the result
dPdth = zeros(nx,nx,nth);

% Compute the Abar matrix, Eq. (5.29)
Abar = A - K*C/dt;

% Compute eigenvectors of Abar
[T,~] = eig(Abar);

% Similarity transformation of Abar,  Eq.(5.31)
Abarprime = T\(Abar*T);

% Loop through all parameters and solve the Lyapunov equation to get dPdth
for pp = 1:nth
    Cbar = -dAdth(:,:,pp)*Pxx + K*dCdth(:,:,pp)*Pxx/dt - dFdth(:,:,pp)*F';
    Qbarprime = -T\((Cbar+Cbar')/(T'));
    dPdthprime = lyap(Abarprime,Qbarprime);
    % E4 = T\((Cbar + Cbar')/(T'));
    % for ii = 1:nx 
    %     for jj = 1:nx 
    %         E5(ii,jj) = E4(ii,jj) / (Abarprime(ii,ii)+Abarprime(jj,jj));    % Eq. (5.33)
    %     end
    % end
    % dPdth(:,:,pp) = T*E5*T';
    dPdth(:,:,pp) = T*dPdthprime*T';
end

end