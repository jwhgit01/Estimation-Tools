classdef squareRootUnscentedKalmanFilterDT < stateEstimatorDT
%squareRootUnscentedKalmanFilterDT
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This class defines the square root unscented Kalman filter for the
% discrete-time nonlinear system
%
%              x(k+1) = f(k,x(k),u(k)) + D(k,x(k),u(k))*v(k)            (1)
%                z(k) = h(k,x(k)) + w(k)                                (2)
%
% where v(k) and w(k) are zero-mean Gaussian white noise with covariances
% Q(k) and R(k), respectively. This filter is implemented and referenced
% by equation number using 
% 
%   van der Merwe & Wan, "The Square-Root Unscented Kalman Filter for State
%       and Parameter-Estimation",  2IEEE International Conference on
%       Acoustics, Speech, and Signal Processing, 2001,
%       https://doi.org/10.1109/ICASSP.2001.940586
% 

properties
    nonlinearDynamics
    measurementModel
    nRK(1,1) {mustBeInteger,mustBePositive} = 10;
    differenceEquation logical
end % public properties

properties (SetAccess=immutable)
    alpha(1,1) double {mustBePositive} = 0.1
    beta(1,1) double {mustBePositive} = 2
    kappa(1,1) double = 0
end % visible immutable properties

properties (SetAccess=immutable,Hidden)
    c(1,1) double
    eta(1,1) double
    Wm(:,1) double
    Wc(:,1) double
    W double
    sqrtW double
    signW0c
end % hidden immutable properties

properties (Access=private)
    nx(1,1) double % number of states
    ns(1,1) double % number of sigma points
end % private properties

methods

    function obj = squareRootUnscentedKalmanFilterDT(f,h,Q,R,differenceEquation,alpha,beta,kappa,nx)
        %squareRootUnscentedKalmanFilterDT Construct an instance of this class

        % Validate inputs
        if ~ischar(f) && ~isstring(f) && ~isa(f,'function_handle')
            error('Input ''nonlindyn'' must be either a char array, string, or function handle')
        end
        if ~ischar(h) && ~isstring(h) && ~isa(h,'function_handle')
            error('Input ''measmodel'' must be either a char array, string, or function handle')
        end

        % Store properties
        obj.nonlinearDynamics = f;
        obj.measurementModel = h;
        obj.processNoiseCovariance = Q;
        obj.measurementNoiseCovariance = R;
        obj.differenceEquation = differenceEquation;
        obj.alpha = alpha;
        obj.beta = beta;
        obj.kappa = kappa;
        obj.nx = nx;
        obj.ns = 2*nx + 1;

        % Compute and store weights
        lambda = obj.alpha^2*(nx+obj.kappa) - nx;
        obj.c = nx + lambda;
        obj.eta = sqrt(obj.c);
        obj.Wm = zeros(2*nx+1,1);
        obj.Wc = zeros(2*nx+1,1);
        obj.Wm(1,1) = lambda/obj.c;
        obj.Wc(1,1) = lambda/obj.c + (1-obj.alpha^2+obj.beta);
        obj.Wm(2:obj.ns,1) = repmat(1/(2*obj.c),2*nx,1);
        obj.Wc(2:obj.ns,1) = repmat(1/(2*obj.c),2*nx,1);
        M = eye(obj.ns) - repmat(obj.Wm,1,obj.ns);
        obj.W = M*diag(obj.Wc)*M';
        obj.sqrtW = (eye(obj.ns) - repmat(obj.Wm,1,obj.ns))*diag(sqrt([abs(obj.Wc(1));obj.Wc(2:obj.ns)]));
        if obj.Wc(1) > 0
            obj.signW0c = "+";
        else
            obj.signW0c = "-";
        end

    end % squareRootUnscentedKalmanFilterDT

    function [xhat,S,nu] = simulate(obj,t,z,u,xhat0,P0,params)
        %simulate This method performs continuous-discrete unscented Kalman
        % filtering for a given time history of measurments.
        %
        % Inputs:
        %
        %   t       The N x 1 sample time vector. The first element of t
        %           corresponds to the time of the initial condition.
        %
        %   z       The N x nz time history of measurements. The first
        %           element of z corresponds to sample k=0, which occurs at
        %           time t=0, and thus is not used.
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

        % Get the problem dimensions and initialize the output arrays.
        N = size(z,1);
        nz = size(z,2);
        xhat = zeros(N,obj.nx);
        S = zeros(obj.nx,obj.nx,N);
        nu = zeros(N,nz);
        xhat(1,:) = xhat0.';
        S(:,:,1) = chol(P0)';
        
        % if no inputs, set to a tall empty array
        if isempty(u)
            u = zeros(N,0);
        end
       

        % This loop performs one model propagation step and one measurement
        % update step per iteration.
        for k = 0:N-2

            % Display the time periodically
            obj.dispIter(t(k+2));

            % Recall, arrays are 1-indexed, but the initial condition
            % occurs at k=0.
            ik = k + 1;
        
            % Perform the dynamic propagation of the state estimate.
            tk = t(ik);
            tkp1 = t(ik+1);
            uk = u(ik,:).';
            xhatk = xhat(ik,:).';
            Sk = S(:,:,ik);
            [xbarkp1,Sbarkp1,Xbarkp1] = obj.predict(k,tk,tkp1,xhatk,uk,Sk,params);
        
            % Perform the measurement update of the state estimate.
            ukp1 = u(ik+1,:).';
            zkp1 = z(ik+1,:).';
            [xhatkp1,Skp1,nukp1] = obj.correct(k+1,zkp1,xbarkp1,ukp1,Sbarkp1,Xbarkp1,params);
            
            % Store the results
            xhat(ik+1,:) = xhatkp1.';
            S(:,:,ik+1) = Skp1;
            nu(ik+1,:) = nukp1.';

        end

    end % simulate

    function [xbarkp1,Sbarkp1,Xbarkp1] = predict(obj,k,tk,tkp1,xhatk,uk,Sk,params)
        %predict State and covatiance propogation step of the UKF

        % Generate sigma points
        Xk = repmat(xhatk,1,obj.ns) + [zeros(obj.nx,1) obj.eta*Sk -obj.eta*Sk];

        % Propogate sigma points through dynamics
        Xbarkp1 = zeros(obj.nx,obj.ns);
        if obj.differenceEquation
            for ii = 1:obj.ns
                Xbarkp1(:,ii) = feval(obj.nonlinearDynamics,tk,Xk(:,ii),uk,[],0,params);
            end
        else
            for ii = 1:obj.ns
                Xbarkp1(:,ii) = c2dNonlinear(Xk(:,ii),uk,[],tk,tkp1,obj.nRK,...
                obj.nonlinearDynamics,0,params);
            end
        end

        % Expected value of the propogated state
        xbarkp1 = Xbarkp1*obj.Wm;

        % Process noise covariance Cholesky factor
        sqrtQ = chol(obj.Q(k))';
        if obj.differenceEquation
            [~,~,Gamk] = feval(obj.nonlinearDynamics,tk,xbarkp1,uk,[],1,params);
        else
            [~,~,Gamk] = c2dNonlinear(xbarkp1,uk,[],tk,tkp1,obj.nRK,...
                obj.nonlinearDynamics,1,params);
        end
        sqrtRv = Gamk*sqrtQ;

        % Use QR factorization and Cholesky update to get the Cholesky
        % factor of the covariance of xbar(k+1)
        dXbarkp1 = Xbarkp1*obj.sqrtW;
        Sbarkp1 = -qr([dXbarkp1(:,2:obj.ns) sqrtRv].',"econ");
        Sbarkp1 = cholupdate(Sbarkp1,dXbarkp1(:,1),obj.signW0c);

        % Debugging check
        % Pbar = Xbarkp1*obj.W*Xbarkp1' + Gamk*obj.Q(k)*Gamk';
        % check = chol(Pbar) - Sbarkp1


    end % predict

    function [xhatkp1,Skp1,nukp1] = correct(obj,kp1,zkp1,xbarkp1,ukp1,Sbarkp1,Xbarkp1,params)
        %predict Measurement correction step of the square-root UKF

        % A priori outputs of sigma points
        Ybar0kp1 = feval(obj.measurementModel,kp1,Xbarkp1(:,1),ukp1,0,params);
        Ybarkp1 = zeros(size(Ybar0kp1,1),obj.ns);
        Ybarkp1(:,1) = Ybar0kp1;
        for ii = 2:obj.ns
            Ybarkp1(:,ii) = feval(obj.measurementModel,kp1,Xbarkp1(:,ii),ukp1,0,params);
        end

        % Weighted average of sigma point outputs
        ybarkp1 = Ybarkp1*obj.Wm;

        % Measurement noise covariance Cholesky factor
        sqrtR = chol(obj.R(kp1))';

        % Square-root computation of the output covariance using
        % Eqs. (24) & (25) in van der Merwe.
        dYbarkp1 = Ybarkp1*obj.sqrtW;
        Sybar = -qr([dYbarkp1(:,2:obj.ns) sqrtR].',"econ");
        Sybar = cholupdate(Sybar,dYbarkp1(:,1),obj.signW0c);

        % Matrix computation of the a priori cross covariance
        Pxybar = Xbarkp1*obj.W*Ybarkp1';

        % Efficient least squares computation of the Kalman gain
        K = (Pxybar/Sybar)/Sybar';

        % LMMSE measurement update
        nukp1 = zkp1 - ybarkp1;
        xhatkp1 = xbarkp1 + K*nukp1;

        % Square-root computation of the a posteriori covariance using
        % Eqs. (28) & (29) in van der Merwe.
        U = K*Sybar';
        Skp1 = Sbarkp1;
        for ii = 1:obj.nx
            Skp1 = cholupdate(Skp1,U(:,ii),"-");
        end

    end % correct

end % public methods

methods (Access=private)

    function dXdt = sigmaPointDynamics(obj,t,X,u,A,params)
        %sigmaPointDynamicsCT This method computes the dynamics of the
        % continuous-time, square-root unscented Kalman filter sigma points
        
        % Dynamics evaluated at each sigma point
        fX = zeros(obj.nx,obj.ns);
        for ii = 1:obj.ns
            fX(:,ii) = feval(obj.nonlinearDynamics,t,X(:,ii),u,[],0,params);
        end

        % Diffusion matrix evaluated at the mean
        [~,~,D] = feval(obj.nonlinearDynamics,t,X(:,1),u,[],1,params);

        % "M" matrix
        M = (A\(X*obj.W*fX' + fX*obj.W*X' + D*obj.Q(t)*D'))/(A');
        
        % Weighed average of dynamics
        fXw = fX*obj.Wm;

        % Sigma point dynamics
        PhiM = obj.Phi(M);
        dXdt = repmat(fXw,1,obj.ns) + sqrt(obj.c)*[zeros(obj.nx,1) A*PhiM -A*PhiM];

    end % sigmaPointDynamics

    function Xk = sigmaPoints(obj,x,S)
        %sigmaPoints Generate matrix of sigma points
        
        % Square root of c
        sqrtc = sqrt(obj.c);
        
        % Construct matrices of sigma points and propogate.
        tic;
        Xk = zeros(obj.nx,obj.ns);
        Xk(:,1) = x;
        for ii = 2:(obj.nx+1)
            Xk(:,ii) = x + sqrtc*S(:,ii-1);
        end
        for ii = (obj.nx+2):obj.ns
            Xk(:,ii) = x - sqrtc*S(:,ii-1-obj.nx);
        end
        toc

        tic;
        Xk = repmat(x,1,obj.ns) + [0 sqrtc*S -sqrtc*S];
        toc
        
    end

end % private methods

methods (Static)
    function Phit = Phi(Mt)
        %Phi
        Phit = 0.5*(triu(Mt) + triu(Mt,1));
    end % Phi
end % static methods

end % classdef