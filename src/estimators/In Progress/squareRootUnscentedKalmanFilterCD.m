classdef squareRootUnscentedKalmanFilterCD < stateEstimatorCD
%squareRootUnscentedKalmanFilterCD
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This class defines the continuous-discrete (hybrid) square root unscented
% Kalman filter for the continuous-time nonlinear system
%
%             dx/dt = f(t,x(t),u(t)) + D(t,x(t),u(t))*vtil(t)           (1)
%
% with discrete measurements
%
%                    z(tk) = h(tk,x(tk)) + w(tk)                        (2)
%
% where vtil(t) is continous-time Gaussian white noise with power spectral
% density Q(t) and w(tk) is zero-mean Gaussian white noise with covariance
% R(tk). This propogation step of this filter is implemented and referenced
% by equation number using 
% 
%   Sarkka, "On Unscented Kalman Filtering for State Estimation of 
%       Continuous-Time Nonlinear Systems", IEEE TRANSACTIONS ON AUTOMATIC
%       CONTROL, VOL. 52, NO. 9, SEPTEMBER 2007, 
%       https://doi.org/10.1109/TAC.2007.904453
% 
% The measurement update step, as recommended by Sarkka, is implemented
% according to
%
%   van der Merwe & Wan, "The Square-Root Unscented Kalman Filter for State
%       and Parameter-Estimation",  2IEEE International Conference on
%       Acoustics, Speech, and Signal Processing, 2001,
%       https://doi.org/10.1109/ICASSP.2001.940586
%
% Properties:
%
%   nonlinearDynamics 
%       The function handle, char array, or string that specifies the
%       function that computes the continuous-time dynamics of the system.
%       The first line of nonlinearDynamics must be in the form
%       [f,A,D] = nonlinearDynamics(t,x,u,vtil,dervflag,params). See
%       nonlinearDynamics_temp.m for details
%
%   measurementModel
%       The function handle, char array, or string that specifies the
%       function that computes the modeled output of the system. The first
%       line of measurementModel must be in the form
%       [h,H] = measurementModel(t,x,u,dervflag,params). See
%       measurementModel_temp.m for details.
%
%   processNoisePSD
%
%   measurementNoiseCovariance
%
%   nRK
%       The number of intermediate Runge-Kutta integrations steps used in
%       the integration of the nonlinear dynamics between measurements.
%
%   alpha
%       A scaling parameter determines the spread of the sigma points about
%       xbar. Typically, one chooses 10e-4 <= alpha <= 1.
%
%   beta
%       A tuning parameter that incorporates information about the prior
%       distribution of x. The value of beta = 2 is optimal for a Gaussian
%       distribution because it optimizes some type of matching of higher
%       order terms (see Wan and van der Merwe).
%
%   kappa
%       A secondary scaling parameter. A good value is typically 3-nx.
%
% Methods
%   unscentedKalmanFilterCD
%   simulate
%   predict
%   correct
%   Q
%   R
%

properties
    nonlinearDynamics
    measurementModel
end % public properties

properties (SetAccess=immutable)
    alpha(1,1) double {mustBePositive} = 0.1
    beta(1,1) double {mustBePositive} = 2
    kappa(1,1) double = 0
end % visible immutable properties

properties (SetAccess=immutable,Hidden)
    c(1,1) double
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

    function obj = squareRootUnscentedKalmanFilterCD(f,h,Q,R,alpha,beta,kappa,nx)
        %unscentedKalmanFilterCD Construct an instance of this class

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
        obj.processNoisePSD = Q;
        obj.measurementNoiseCovariance = R;
        obj.alpha = alpha;
        obj.beta = beta;
        obj.kappa = kappa;
        obj.nx = nx;
        obj.ns = 2*nx + 1;

        % Compute and store weights
        lambda = obj.alpha^2*(nx+obj.kappa) - nx;
        obj.c = nx + lambda;
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

    end % unscentedKalmanFilterCD

    function [xhat,A,nu] = simulate(obj,t,z,u,xhat0,P0,params)
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
        A = zeros(obj.nx,obj.nx,N);
        nu = zeros(N,nz);
        xhat(1,:) = xhat0.';
        A(:,:,1) = chol(P0)';
        
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
            Ak = A(:,:,ik);
            [xbarkp1,Abarkp1,Xbarkp1] = obj.predict(tk,tkp1,xhatk,uk,Ak,params);
        
            % Perform the measurement update of the state estimate.
            ukp1 = u(ik+1,:).';
            zkp1 = z(ik+1,:).';
            [xhatkp1,Akp1,nukp1] = obj.correct(k+1,zkp1,xbarkp1,ukp1,Abarkp1,Xbarkp1,params);
            
            % Store the results
            xhat(ik+1,:) = xhatkp1.';
            A(:,:,ik+1) = Akp1;
            nu(ik+1,:) = nukp1.';

        end

    end % simulate

    function [xbarkp1,Abarkp1,Xbarkp1] = predict(obj,tk,tkp1,xhatk,uk,Ak,params)
        %predict State propogation step of the UKF

        % Generate sigma points
        Xk = obj.sigmaPoints(tk,xhatk,uk,Ak,params);
        
        % Prepare for the Runge-Kutta numerical integration by setting up 
        % the initial conditions and the time step.
        X = Xk;
        t = tk;
        delt = (tkp1-tk)/obj.nRK;
        
        % This loop does one 4th-order Runge-Kutta numerical integration
        % step per iteration.  Integrate the state.  If partial derivatives
        % are to be calculated, then the partial derivative matrices
        % simultaneously with the state.
        for jj = 1:obj.nRK
        
            % Step a
            dXdt = obj.sigmaPointDynamics(t,X,uk,Ak,params);
            dXa = dXdt*delt;
        
            % Step b
            dXdt = obj.sigmaPointDynamics(t+0.5*delt,X+0.5*dXa,uk,Ak,params);
            dXb = dXdt*delt;
        
            % Step c
            dXdt = obj.sigmaPointDynamics(t+0.5*delt,X+0.5*dXb,uk,Ak,params);
            dXc = dXdt*delt;
        
            % Step d
            dXdt = obj.sigmaPointDynamics(t+delt,X+dXc,uk,Ak,params);
            dXd = dXdt*delt;
        
            % 4th order Runge-Kutta integration result
            X = X + (dXa + 2*(dXb + dXc) + dXd)/6;
            t = t + delt;
        
        end
        
        % Assign the results to the appropriate outputs.
        Xbarkp1 = X;

        % Extract the Cholesky factor of the covariance of the propogated
        % state from the sigma points.
        xbarkp1 = Xbarkp1*obj.Wm;
        % Abarkp1 = triu((Xbarkp1(:,2:obj.nx+1) - repmat(xbarkp1,1,obj.nx))/sqrt(obj.c));

        % Use QR factorization and Cholesky update to get the Cholesky
        % factor of the covariance
        sqrtQ = chol(obj.Q(tkp1))';
        [~,~,Dtkp1] = feval(obj.nonlinearDynamics,tkp1,xbarkp1,uk,[],1,params);
        dXbarkp1 = Xbarkp1*obj.sqrtW;
        Abarkp1 = -qr([Dtkp1*sqrtQ dXbarkp1(:,2:obj.ns)].',"econ");
        Abarkp1 = cholupdate(Abarkp1,dXbarkp1(:,1),obj.signW0c);

    end % predict

    function [xhatkp1,Akp1,nukp1] = correct(obj,kp1,zkp1,xbarkp1,ukp1,Abarkp1,Xbarkp1,params)
        %predict Measurement correction step of the square-root UKF based
        % on van der Merwe & Wan.

        % A priori outputs of sigma points
        Ybar0kp1 = feval(obj.measurementModel,kp1,Xbarkp1(:,1),ukp1,0,params);
        Ybarkp1 = zeros(size(Ybar0kp1,1),obj.ns);
        Ybarkp1(:,1) = Ybar0kp1;
        for ii = 2:obj.ns
            Ybarkp1(:,ii) = feval(obj.measurementModel,kp1,Xbarkp1(:,ii),ukp1,0,params);
        end

        % Weighted average of sigma point outputs
        ybarkp1 = Ybarkp1*obj.Wm;

        % Square-root computation of the output covariance using
        % Eqs. (24) & (25) in van der Merwe.
        sqrtR = chol(obj.R(kp1))';
        dYbarkp1 = Ybarkp1*obj.sqrtW;
        Sybar = -qr([sqrtR dYbarkp1(:,2:obj.ns)].',"econ");
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
        Akp1 = Abarkp1;
        for ii = 1:obj.nx
            Akp1 = cholupdate(Akp1,U(:,ii),"-");
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

    function [X,F] = sigmaPoints(obj,t,x,u,A,params)
        %sigmaPoints This method computes matrices of sigma points
        % according to Eq. (32) along with their dynamics.
        
        % Evaluate the dynamics given state, x.
        fX0 = feval(obj.nonlinearDynamics,t,x,u,[],0,params);
        
        % Square root of c
        sqrtc = sqrt(obj.c);
        
        % Construct matrices of sigma points and their dynamics.
        X = zeros(obj.nx,obj.ns);
        X(:,1) = x;
        F = zeros(obj.nx,obj.ns);
        F(:,1) = fX0;
        for ii = 2:(obj.nx+1)
            X(:,ii) = x + sqrtc*A(:,ii-1);
            F(:,ii) = feval(obj.nonlinearDynamics,t,X(:,ii),u,[],0,params);
        end
        for ii = (obj.nx+2):obj.ns
            X(:,ii) = x - sqrtc*A(:,ii-1-obj.nx);
            F(:,ii) = feval(obj.nonlinearDynamics,t,X(:,ii),u,[],0,params);
        end
        
    end

end % private methods

methods (Static)
    function Phit = Phi(Mt)
        %Phi
        Phit = 0.5*(triu(Mt) + triu(Mt,1));
    end % Phi
end % static methods

end % classdef