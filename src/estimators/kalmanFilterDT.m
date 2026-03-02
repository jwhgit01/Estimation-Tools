classdef kalmanFilterDT < stateEstimatorDT
%kalmanFilterDT
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This class defines the discrete-time Kalman filter for the discrete-time
% linear system
%
%               x(k+1) = F(k)*x(k) + G(k)*u(k) + Gamma(k)*w(k)          (1)
%
% where u(k) is a known input and each w(k) is independently samped from a
% Gaussian distribtion with zero mean and covariance Q(k). The linear
% measurement model for this system is
%
%                 z(k) = C(k)*x(k) + D(k)*u + v(k)                      (2)
%
% where each v(k) is independently samped from a Gaussian distribtion with
% zero mean and covariance R(k).
%
% The inputs to the constructor of this class are:
%
% F     The state transition matrix F in Eq.(1). It may be specified as
%       a contant matrix, a nx x nx x N array, or function handle that
%       returns an nx x nx matrix given the sample number k.
%
% G     The input matrix G in Eq.(1). It may be specified as a contant
%       matrix, a nx x nu x N array, or function handle that returns an
%       nx x nu matrix given the sample number k. If there is no input
%       to the system, G may be set as an empty array, [].
%
% Gamma The noise dispersion matrix Gamma in Eq.(1). It may be
%       specified as a contant matrix, a nx x nw x N array, or function
%       handle that returns an nx x nw matrix given the sample number
%       k. If the noise enters in every state equation (that is,
%       Gamma = eye(nx)), then Gamma may be set as an empty array, [].
%
% C     The measurement matrix C in Eq.(2). It may be specified as a
%       contant matrix, a nz x nx x N array, or function handle that
%       returns an nz x nx matrix given the sample number k.
%
% D     The throughput matrix D in Eq.(2). It may be specified as a
%       contant matrix, a nz x nu x N array, or function handle that
%       returns an nz x nu matrix given the sample number k. If there
%       is no inpu t to the system, or if there is no throughput term,
%       D may be set as an emptyarray, [].
%
%
% Properties
%   DisplayIteration
%
% Methods
%   kalmanFilterDT
%   simulate
%   smooth (TODO)
%
properties
    % No additonal 
end % public properties

properties (Access=private)
    Fh
    Gh
    Gamh
    Ch
    Dh
    Qh
    Rh
    nx
    nu
    nw
    nz
end % private properties

methods

    function obj = kalmanFilterDT(F,G,Gamma,C,D,Q,R)
        %kalmanFilterDT Construct an instance of this class.
    
        % State transition matrix and number of states
        obj.Fh = obj.makeFun(F);
        nx = size(obj.Fh(0),1);

        % Output matrix and number of outputs
        obj.Ch = obj.makeFun(C);
        nz = size(obj.Ch(0),1);

        % Dispersion matrix and number of disturbances
        if isempty(Gamma)
            Gamma = eye(nx);
        end
        obj.Gamh = obj.makeFun(Gamma);
        nw = size(obj.Gamh(0),2);

        % Input matrix and number of inputs
        if isempty(G)
            G = zeros(nx,0);
        end
        obj.Gh = obj.makeFun(G);
        nu = size(obj.Gh(0),2);

        % Throughput matrix
        if isempty(D)
            D = zeros(nz,nu);
        end
        obj.Dh = obj.makeFun(D);

        % Process and measurement noise covariances
        obj.Qh = obj.makeFun(Q);
        obj.Rh = obj.makeFun(R);

        % Store dimensions
        obj.nx = nx;
        obj.nu = nu;
        obj.nw = nw;
        obj.nz = nz;

    end % kalmanFilterDT

    function [xhat,P,nu,epsnu,sigdig] = simulate(obj,z,u,xhat0,P0)
        %simulate This method performs discrete-time Kalman filtering for a
        % given time history of measurments.
        %
        % Inputs:
        %
        %   z       The N x nz time history of measurements. The first
        %           element of z corresponds to sample k=0, and thus is not
        %           used.
        %
        %   u       The N x nu time history of system inputs (optional). If
        %           not applicable set to an empty array, []. The first
        %           element of u corresponds to the input at the initial
        %           condition.
        %
        %   xhat0   The nx x 1 initial state estimate.
        %
        %   P0      The nx x nx symmetric positive definite initial state
        %           estimation error covariance matrix.
        %
        %   params  An array or struct of constants that get passed to the
        %           dynamics model and measurement model functions.
        %  
        % Outputs:
        %
        %   xhat    The N x nx array that contains the time history of the
        %           state vector estimates.
        %
        %   P       The nx x nx x N array that contains the time history of
        %           the estimation error covariance.
        %
        %   nu      The N x nz vector of innovations.
        %
        %   epsnu   The N x 1 vector of the normalized innovation statistic
        %
        %   sigdig  The approximate number of accurate significant decimal
        %           places in the result. This is computed using the
        %           condition number of inovation covariance, S.
        %

        % Initialization
        N = size(z,1); % number of samples (including initial conditon)
        xhat = zeros(N,obj.nx); % state estimate
        P = zeros(obj.nx,obj.nx,N); % state estimate covariance
        nu = nan(N,obj.nz); % innovation
        epsnu = nan(N,1); % innovation statistic
        xhat(1,:) = xhat0.'; % initial estimate
        P(:,:,1) = P0; % initial estimate covariance
        maxsigdig = -fix(log10(eps)); % accuracy based on machine precision
        sigdig = maxsigdig; % initial accuracy
        
        % If no inputs, set to a tall empty array
        if isempty(u)
            u = zeros(N,0);
        end
        
        % This loop performs one model propagation step and one measurement
        % update step per iteration.
        for k = 0:N-2

            % Display the iteration number periodically
            obj.dispIter(k);

            % Get system matrices
            Fk = obj.Fh(k);
            Gk = obj.Gh(k);
            Gamk = obj.Gamh(k);
            Qk = obj.Qh(k);
            Ckp1 = obj.Ch(k+1);
            Dkp1 = obj.Dh(k+1);
            Rkp1 = obj.Rh(k+1);

            % Recall, arrays are 1-indexed, but the initial condition
            % occurs at k=0. Use ik to index arrays.
            ik = k + 1;

            % Propagate the state estimate to the next time step.
            xhatk = xhat(ik,:).'; % state estimate at k
            uk = u(ik,:).'; % input at k
            Pk = P(:,:,ik); % covariance of state estimate at k
            xbarkp1 = Fk*xhatk + Gk*uk; % state prediction at k+1
            Pbarkp1 = Fk*Pk*Fk' + Gamk*Qk*Gamk'; % covariance of prediction

            % Compute the innovation and its covariance.
            ukp1 = u(ik+1,:).'; % input at k+1
            zkp1 = z(ik+1,:).'; % measurement at k+1
            ybarkp1 = Ckp1*xbarkp1 + Dkp1*ukp1; % predicted output
            nukp1 = zkp1 - ybarkp1; % innovation
            Skp1 = Ckp1*Pbarkp1*Ckp1' + Rkp1; % innovation covariance

            % Perform the measurement update.
            Wkp1 = (Pbarkp1*Ckp1')/Skp1; % Kalman gain
            xhatkp1 = xbarkp1 + Wkp1*nukp1; % state estimate
            Pkp1 = Pbarkp1 - Wkp1*Skp1*Wkp1'; % stat estimate covariance
            
            % Store results
            xhat(ik+1,:) = xhatkp1.';
            P(:,:,ik+1) = Pkp1;
            nu(ik+1,:) = nukp1.';
            epsnu(ik+1,:) = nukp1'*(Skp1\nukp1);
        
            % Check the condition number of Skp1 and infer the approximate
            % numerical precision of the resulting estimate.
            sigdigkp1 = maxsigdig - fix(log10(cond(Skp1)));
            if sigdigkp1 < sigdig
                sigdig = sigdigkp1;
            end
        
        end

    end % simulate

end % public methods

end % classdef