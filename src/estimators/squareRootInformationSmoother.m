function [xs,Pxxs,vs,Pvvs,Pvxs] = squareRootInformationSmoother(z,u,F,G,Gam,H,Q,R,xhat0,I0,ivs)
%squareRootInformationSmoother 
%
%  Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs linear square root information Rauch-Tung-Striebel
% (RTS) smoothing time history of measurments and the discrete-time linear
% system,
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
%   ivs     A boolean dictating whether to compute and return a smoothed
%           estimate of the process nosie, along with its covariance and
%           cross-covariance with x.
%  
% Outputs:
%
%   xs      The N x nx array that contains the time history of the
%           smoothed state vector estimates.
%
%   Pxxs    The nx x nx x N array that contains the time history of the
%           covariance matrices of the smoothed estimates.
%
%   vs      The N x nv array that contains the time history of the smoothed
%           estimate of the process noise.
% 
%   Pvvs    The nv x nv x N array that contains the time history of the
%           covariance matrices of the estimated process noise.
% 
%   Pvxs    The nv x nx x N array that contains the time history of the
%           cross-covariance matrices of the estimated process noise with
%           the state vector.
%

% Check to see whether we have a time-varying or time-invariant system. A
% time-varying system may be prescribed by an array of matrices or a
% function handle that is a fucntion of the time step k. If an input is a
% constant matrix or a 3-dim array, make it an anonymous function.
if ~isa(F,'function_handle')
    if size(F,3) > 1, Fk = @(k) F(:,:,k+1); else, Fk = @(k) F; end
end
if ~isempty(G) && ~isa(G,'function_handle')
    if size(G,3) > 1, Gk = @(k) G(:,:,k+1); else, Gk = @(k) G; end
end
if ~isa(Gam,'function_handle')
    if size(Gam,3) > 1, Gamk = @(k) Gam(:,:,k+1); else, Gamk = @(k) Gam; end
end
if ~isa(Q,'function_handle')
    if size(Q,3) > 1, Qk = @(k) Q(:,:,k+1); else, Qk = @(k) Q; end
end

% Get the problem dimensions.
N = size(z,1);
nx = size(xhat0,1);
nv = size(Qk(0),1);

% Perform square root information filtering forward in time.
[xhat,Phat,Rscrvvbar,Rscrvxbar,zscrvbar] = ...
                   squareRootInformationFilter(z,u,F,G,Gam,H,Q,R,xhat0,I0);

% Initialize the smoothed estimate, covariance, and square root information
% matrix & output.
xs = xhat;
Pxxs = Phat;
Rscrxxskp1 = chol(inv(Phat(:,:,N)));
zscrxskp1 = Rscrxxskp1*(xhat(N,:).');

% If desired, initialize the smoothed nosie estimate and its covariances.
if ivs
    vs = zeros(N,nv);
    Pvvs = zeros(nv,nv,N);
    Pvxs = zeros(nv,nx,N);
else
    vs = [];
    Pvvs = [];
    Pvxs = [];
end

% This loop performs one smoothing step backwards in time per iteration.
for k = N-1:-1:0

    % Recall, arrays are 1-indexed, but the initial condition occurs at k=0
    kp1 = k+1;
    
    % Use QR factorization to compute Rscrvv(k), Rscrvx(k), Rscrxx(k), and
    % Tc(k).
    A11 = Rscrvvbar(:,:,kp1)+Rscrvxbar(:,:,kp1+1)*Gamk(k);
    A12 = Rscrvxbar(:,:,kp1+1)*Fk(k);
    A21 = Rscrxxskp1*Gamk(k);
    A22 = Rscrxxskp1*Fk(k);
    [Tcktr,Rscrs] = qr([A11,A12;A21,A22]);
    Rscrvvsk = Rscrs(1:nv,1:nv);
    Rscrvxsk = Rscrs(1:nv,nv+1:nv+nx);
    Rscrxxsk = Rscrs(nv+1:nv+nx,nv+1:nv+nx);
    
    % Compute the smoothed square root information outputs.
    if isempty(u) || isempty(G)
        zscrsk = Tcktr'*[zscrvbar(kp1,:).';zscrxskp1];
    else
        zscrsk = Tcktr'*[zscrvbar(kp1,:).'-Rscrvxbar(:,:,kp1+1)*Gk(k)*u(kp1,:).';...
                         zscrxskp1-Rscrxxskp1*Gk(k)*u(kp1,:).'];
    end
    zscrvsk = zscrsk(1:nv,:);
    zscrxsk = zscrsk(nv+1:nv+nx,:); 

    % Compute and store the smoothed state estimate and its covariance.
    Rscrxxskinv = inv(Rscrxxsk);
    xs(kp1,:) = (Rscrxxsk\zscrxsk).';
    Pxxs(:,:,kp1) = Rscrxxskinv*Rscrxxskinv';

    % If desired, compute the smoothed noise estimate along with its
    % covariance and cross-covariance with x.
    if ivs
        vs(kp1,:) = Rscrvvsk\(zscrvsk-Rscrvxsk*xs(kp1,:).');
        Pvvs(:,:,kp1) = Rscrvvsk\(eye(nv)+Rscrvxsk*Pxxs(:,:,kp1)*Rscrvxsk')/(Rscrvvsk');
        Pvxs(:,:,kp1) = -(Rscrvvsk\Rscrvxsk)*Pxxs(:,:,kp1);
    end

    % Update zscrxs(k+1) and Rscrxxs(k+1) for next iteration.
    zscrxskp1 = zscrxsk;
    Rscrxxskp1 = Rscrxxsk;

end

end