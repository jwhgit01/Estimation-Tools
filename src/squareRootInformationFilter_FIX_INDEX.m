function [xhat,P,Rscrvvbar,Rscrvxbar,zscrvbar] = squareRootInformationFilter(z,u,F,G,Gam,H,Q,R,xhat0,I0)
%squareRootInformationFilter 
%
%  Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs linear square root information filtering for a
% given time history of measurments and the discrete-time linear system,
%
%   x(k+1) = F(k)*x(k) + G(k)*u(k) + Gam(k)*v(k)                    (1)
%     z(k) = H(k)*x(k) + w(k)                                       (2)
%
% where v(k) is zero-mean Gaussian, white noise with covariance Q(k) and
% w(k) is zero-mean Gaussian, white noise with covariance R(k).
%
% Inputs:
%
%   z       The N x nz time history of measurements.
%
%   u       The N x nu time history of system inputs (optional). If not
%           applicable set to zeros or an empty array, [].
% 
%   F,G,Gam The system matrices in Eq.(1). These may be specified as
%           contant matrices, (.)x(.)xN arrays of matrices, or function
%           handles that return a matrix given the time step k. Note that
%           if there is no input, G may be given as an empty array, [].
% 
%   H       The measurement model matrix in Eq.(2). This may be specified
%           as a contant matrix, an (.)x(.)xN array, or a function handle
%           that returns a matrix given the time step k.
%   
%   Q,R     The process and measurement noise covariance. These may be
%           specified as contant matrices, (.)x(.)xN arrays of matrices, or
%           function handles that returns a matrix given the time step k.
%
%   xhat0   The nx x 1 initial state estimate.
%
%   I0      The nx x nx symmetric positive semi-definite initial state
%           estimation information matrix. It is the inverse of the initial
%           state estimate error covariance matrix.
%  
% Outputs:
%
%   xhat        The N x nx array that contains the time history of the
%               state vector estimates.
%
%   P           The nx x nx x N array that contains the time history of the
%               estimation error covariance matrices.
%
%   Rscrvvbar
% 
%   Rscrvxbar
% 
%   zscrvbar
%

% Check to see whether we have a time-varying or time-invariant system. A
% time-varying system may be prescribed by an array of matrices or a
% function handle that is a fucntion of the time step k. If an input is a
% constant matrix or a 3-dim array, make it an anonymous function.
if ~isa(F,'function_handle')
    if size(F,3) > 1, Fk = @(k) F(:,:,k); else, Fk = @(k) F; end
end
if ~isempty(G) && ~isa(G,'function_handle')
    if size(G,3) > 1, Gk = @(k) F(:,:,k); else, Gk = @(k) G; end
end
if ~isa(Gam,'function_handle')
    if size(Gam,3) > 1, Gamk = @(k) Gam(:,:,k); else, Gamk = @(k) Gam; end
end
if ~isa(H,'function_handle')
    if size(H,3) > 1, Hk = @(k) H(:,:,k); else, Hk = @(k) H; end
end
if ~isa(Q,'function_handle')
    if size(Q,3) > 1, Qk = @(k) Q(:,:,k); else, Qk = @(k) Q; end
end
if ~isa(R,'function_handle')
    if size(R,3) > 1, Rk = @(k) R(:,:,k); else, Rk = @(k) R; end
end

% Get the problem dimensions and initialize the output arrays.
N = size(z,1);
nx = size(xhat0,1);
% nz = size(z,2);
nv = size(Qk(1),1);
xhat = zeros(N,nx);
P = zeros(nx,nx,N);
xhat(1,:) = xhat0.';
P(:,:,1) = inv(I0);
Rscrvvbar = zeros(nv,nv,N);
Rscrvxbar = zeros(nv,nx,N);
zscrvbar = zeros(N,nv);

% Compute the initial square-root information output and matrix
Rscrxxk = chol(I0);
zscrxk = Rscrxxk*xhat0;

% This loop performs one model propagation step and one measurement
% update step per iteration.
for k = 1:N-1
    
    % Compute the Cholesky factor of the process noise at sample k.
    Rscrvvk = inv(chol(Qk(k)))';

    % Compute a priori information matrices, Rscfvvbar(k), Rscvxbar(k+1),
    % and Rscxxbar(k+1) via QR factorization.
    [Taktr,Rscrbar] = qr([Rscrvvk,zeros(nv,nx); ...
                         -(Rscrxxk/Fk(k))*Gamk(k),Rscrxxk/Fk(k)]);
    Rscrvvbar(:,:,k) = Rscrbar(1:nv,1:nv);
    Rscrvxbar(:,:,k+1) = Rscrbar(1:nv,nv+1:nv+nx);
    Rscxxbarkp1 = Rscrbar(nv+1:nv+nx,nv+1:nv+nx);

    % Propogate the SQIF outputs using Ta(k).
    if isempty(u) || isempty(G)
        zscrbar = Taktr'*[zeros(nv,1);zscrxk];
    else
        zscrbar = Taktr'*[zeros(nv,1);zscrxk+(Rscrxxk/Fk(k))*Gk(k)*u(k,:).'];
    end
    zscrvbar(k,:) = zscrbar(1:nv,:).';
    zscrxbarkp1 = zscrbar(nv+1:nv+nx,:);

    % Compute the Cholesky factor of the measurement noise covariance for
    % sample k+1.
    Rakp1tr = chol(Rk(k+1))';

    % Compute the square-root measurement matrix and measurement.
    Hakp1 = Rakp1tr\Hk(k+1);
    zakp1 = Rakp1tr\(z(k+1,:).');
    
    % Compute the a posteriori state information matrix, Rscrxxkp1.
    [Tbkp1tr,Rscr] = qr([Rscxxbarkp1;Hakp1]);
    Rscrxxkp1 = Rscr(1:nx,:);

    % Perform the square-rooot information measurement update.
    zscrkp1 = Tbkp1tr'*[zscrxbarkp1;zakp1];
    zscrxkp1 = zscrkp1(1:nx,:);
    % zscrrkp1 = zscrkp1(nx+1:nx+nz,:);

    % Compute and store the state estimate and its covariance.
    Rscrxxkp1inv = inv(Rscrxxkp1);
    xhat(k+1,:) = (Rscrxxkp1\zscrxkp1).';
    P(:,:,k+1) = Rscrxxkp1inv*Rscrxxkp1inv';

    % Update the square root information output and matrix.
    Rscrxxk = Rscrxxkp1;
    zscrxk = zscrxkp1;

end

end