function [xhat,P,Rscrvvbar,Rscrvxbar,zscrvbar] = squareRootInformationEKF(t,z,u,f,h,Q,R,xhat0,I0,nRK,params)
%squareRootInformationEKF 
%
%  Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs extended square root information filtering for a
% given time history of measurments and the discrete-time nonlinear system,
%
%                   x(k+1) = f(k,x(k),u(k),v(k))                (1)
%                     z(k) = k(k,x(k),w(k))                     (2)
%
% where v(k) is zero-mean Gaussian, white noise with covariance Q(k) and
% w(k) is zero-mean Gaussian, white noise with covariance R(k).
%
% Inputs:
%
%   t       The Nz x 1 sample time vector. If f is a discrete-time dynamic
%           model, t must be givenn as an empty array, [].
%
%   z       The N x nz time history of measurements.
%
%   u       The N x nu time history of system inputs (optional). If not
%           applicable set to zeros or an empty array, [].
% 
%   f       The function handle that computes either the continuous-time
%           dynamics if t is given as a vector of sample times or the
%           discrete-time dynamics if t is empty. The first line of f must
%           be in the form
%               [f,A,D] = nonlindyn(t,x,u,vtil,dervflag,params)
%           or
%               [fk,Fk,Gamk] = nonlindyn(k,xk,uk,vk,dervflag,params)
%
%   h       ...
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
%   nRK     ...
%
%   params  ...
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

% Check to see whether we have non-stationary noise, which may
% be prescribed by an array of matrices or a function handle that is a
% fucntion of the timestep/time.
if ~isa(R,'function_handle')
    if size(R,3) > 1, Rk = @(k) R(:,:,k); else, Rk = @(k) R; end
else
    Rk = R;
end
if ~isa(Q,'function_handle')
    if size(Q,3) > 1, Qk = @(k) Q(:,:,k); else, Qk = @(k) Q; end
else
    Qk = Q;
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

    disp(k);
    
    % Compute the Cholesky factor of the process noise at sample k.
    Rscrvvk = inv(chol(Qk(k)))';

    % Linearize the dynamics about xhat(k).
    xhatk = xhat(k,:).';
    uk = u(k,:).';
    if isempty(t)
        [xbarkp1,Fk,Gamk] = feval(f,k,xhatk,uk,zeros(nv,1),1,params);
    else
        [xbarkp1,Fk,Gamk] = c2dNonlinear(xhatk,uk,zeros(nv,1),t(k,1),t(k+1,1),nRK,f,1,params);
    end

    % Compute a priori information matrices, Rscfvvbar(k), Rscvxbar(k+1),
    % and Rscxxbar(k+1) via QR factorization.
    [Taktr,Rscrbar] = qr([Rscrvvk,zeros(nv,nx); ...
                         -(Rscrxxk/Fk(k))*Gamk,Rscrxxk/Fk]);
    Rscrvvbar(:,:,k) = Rscrbar(1:nv,1:nv);
    Rscrvxbar(:,:,k+1) = Rscrbar(1:nv,nv+1:nv+nx);
    Rscxxbarkp1 = Rscrbar(nv+1:nv+nx,nv+1:nv+nx);

    % Propogate the SQIF outputs using Ta(k).
    if isempty(u)
        zscrbar = Taktr'*[zeros(nv,1);zscrxk];
    else
        zscrbar = Taktr'*[zeros(nv,1);zscrxk+(Rscrxxk/Fk)*Gk(k)*u(k,:).'];
    end
    zscrvbar(k,:) = zscrbar(1:nv,:).';
    zscrxbarkp1 = zscrbar(nv+1:nv+nx,:);

    % Compute the Cholesky factor of the measurement noise covariance for
    % sample k+1.
    Rakp1tr = chol(Rk(k+1))';

    % Linearize the measurment model about xbar(k+1).
    ukp1 = u(k+1,:).';
    [~,Hkp1] = feval(h,t(k+1),xbarkp1,ukp1,1,0,params);

    % Compute the square-root measurement matrix and measurement.
    Hakp1 = Rakp1tr\Hkp1;
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