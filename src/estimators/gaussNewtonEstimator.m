classdef gaussNewtonEstimator
%gaussNewtonEstimator
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This class implements nonlinear batch least-squares estimation using
% the Gauss-Newton method for the nonlinear model
% 
%                     z(k) = h(k,x(k),u(k)) + v(k)                      (1)
%
% where x is the value to be estimated, u is an array of known inputs, and
% each v(k) is independently sampled from a Gaussian distribution with zero
% mean and covariance R(k). It aims to minimize the cost function
%
%              N  /              T     -1                \
%      J[x] = SUM | [z(i)-h(i,x)]   R(i)   [z(i)-h(i,x)] |              (2)
%             i=1 \                                      /
%
% Properties:
%   MeasurementModel
%   MeasurementNoiseCovariance
%
% Methods
%   gaussNewtonEstimator
%   estimate
%   cost
%

properties
    % The function handle that defines the measurment model (1). It must be
    % in the form [y,H] = meas(k,x,u,params), where y is the modeleed
    % output and H is its Jacobian. Here, k is the sample number, x is the
    % state/parameter vector, u is the input vector, and params is a struct
    % of constant parameters.
    MeasurementModel

    % The measurement noise covariance, which may be a constant matrix, an
    % nz x nz x N 3-dimensional array, or the handle of a function whose
    % input is the sample number, k.
    MeasurementNoiseCovariance

    % A logical indicating whether information about the Gauss-Newton
    % iterations should be displayed.
    DisplayIterations logical = true

    % The tolerance factor on dJ such that Gauss-Newton iterations
    % terminate if abs(dJ)<dJTol*(1+J) and norm(dx)<dxTol*(1+norm(xguess)).
    dJTol (1,1) {mustBePositive} = 1e-12

    % The tolerance factor on dx such that Gauss-Newton iterations
    % terminate if abs(dJ)<dJTol*(1+J) and norm(dx)<dxTol*(1+norm(xguess)).
    dxTol (1,1) {mustBePositive} = 1e-9

    % The tolerance on the ratio of changes in cost function value such
    % that if The change in x is sufficiently small, the ratio of changes
    % in delta J is sufficiently close to 1 for the case where the step
    % size has not been halved.
    dJratioTol (1,1) {mustBePositive} = 1e-2

    % The maximum number of Gauss-Newton iterations
    maxIterations (1,1) {mustBeInteger,mustBePositive} = 100;
end % public properties

properties (SetAccess=immutable,Hidden)
    invCholR
end % hidden immutable properties

properties (Access=private)
    %
end % private properties

methods
    function obj = gaussNewtonEstimator(h,R)
        %gaussNewtonEstimator Construct an instance of this class

        % Empty object
        if nargin < 1
            return
        end

        % Store properties
        obj.MeasurementModel = h;
        obj.MeasurementNoiseCovariance = R;

        % Inverse of the Cholesky factor of R
        if ~isa(R,'function_handle') && size(R,3) == 1
            obj.invCholR = inv(chol(R));
        end

    end % gaussNewtonEstimator

    function [xhat,Jopt,P,termflag] = estimate(obj,z,u,xguess,params)
        % This function performs nonlinear batch least-squares estimation
        % using the Gauss-Newton method.
        %
        % Inputs:
        %
        %   z       The N x nx time history of measurements
        %
        %   u       The N x nu time history of inputs/regresors
        %
        %   xguess  The initial guess of x
        %
        %   params  A struct of constant parameters for the model
        %  
        %  Outputs:
        %
        %   xhat        The final estimate of 
        %
        %   Jopt        The final optimal value of J given in (2)
        %
        %   P           The Gauss-Newton approximation of the estimation
        %               error covariance.
        %
        %   termflag    A termination flag that indicates whether the
        %               procedure terminated well or not.
        %
        %                   0   Normal termination at the optimum
        %
        %                   1   Terminated because more than 50 step
        %                       size halvings were required. This
        %                       may be an optimum.
        %
        %                   2   Terminated because more than 100
        %                       Gauss-Newton iterations were required.
        %                       This may not be the optimum.
        %
        
        % Calculate the initial cost and Gauss-Newton search direction
        [J,dx,dJdalpha,d2Jdalpha2,P] = obj.cost(xguess,z,u,params);
        
        % Predict the change in cost if a step size of alpha = 1 is taken
        dJpredict = dJdalpha + 0.5*d2Jdalpha2;

        % Decide whether to terminate at first step
        dJTest = abs(dJpredict) < obj.dJTol*(1 + J);
        dxTest = norm(dx) < obj.dxTol*(1 + norm(xguess));
        termflag = 0;
        if dJTest && dxTest
            xhat = xguess;
            Jopt = J;
            return
        end
        
        % Initialize Gauss-Newton iteration variables
        converge = false;
        iteration = 0;
        stepFlag = 0;
        
        % Perform one Gauss-Newton iteration per loop
        while ~converge
            
            % Initial step size and Gauss-Newton iteration
            alpha = 1;
            xguessnew = xguess + alpha*dx;
            Jnew = obj.cost(xguessnew,z,u,params);
        
            % Half the step size if necessary in order to force a decrease
            % in the cost.
            nalphahalf = 0;
            while Jnew >= J
                nalphahalf = nalphahalf + 1;
                if nalphahalf > 50
                    stepFlag = 1;
                    break
                end
                alpha = 0.5*alpha;
                xguessnew = xguess + alpha*dx;
                Jnew = obj.cost(xguessnew,z,u,params);
            end
            
            % If we have halved the step size more than 50 times, make note
            if stepFlag == 1
                termflag = 1;
                break
            end
        
            % Update variables for next iteration
            xguess = xguessnew;
            Jold = J;
            dJold = Jnew - J;
            dJpredictOld = dJpredict;
        
            % Perform a Gauss-Newton iteration
            [J,dx,dJdalpha,d2Jdalpha2,P] = obj.cost(xguess,z,u,params);
        
            % Convergence criteria
            dJpredict = dJdalpha + 0.5*d2Jdalpha2;
            dJTest = abs(dJpredict) < obj.dJTol*(1 + J);
            dxTest = norm(dx) < obj.dxTol*(1 + norm(xguess));
            alphaTest = alpha == 1;
            delJratiotest = abs((dJold/dJpredictOld) - 1) < obj.dJratioTol;
           
            % Criteria #1: The changes in x and J are sufficiently small. 
            if dJTest && dxTest
                converge = true;
            end
        
            % Criteria #2: The change in x is sufficiently small, the ratio
            %              of changes in delta J is sufficiently close to 1
            %              for the case where the step size has not been
            %              halved.
            if alphaTest && delJratiotest && dxTest
                converge = true;
            end
        
            % If we have arrived here, we have not converged nicely to a
            % optimal value. Perform more Gauss-Newton iterations.
            iteration = iteration + 1;
        
            % Criteria #3: We have succeeded the maximum number of
            %              Gauss-Newton iterations.
            if ~converge && iteration >= obj.maxIterations
                termflag = 2;
                converge = true;
            end
        
            % If desired, display the results of this Gauss-Newton iteration.
            if obj.DisplayIterations
                fprintf('Iteration %02i : alpha = %06.4e, Jnew = %06.4e, Jold = %06.4e, norm(delxnew) = %06.4e\n',...
                    iteration,alpha,J,Jold,norm(dx));
            end

        end % Gauss-Newton iteration loop
        
        % Return the final results
        xhat = xguess;
        Jopt = J;

    end

    function [J,dx,dJdalpha,d2Jdalpha2,P,dJdx] = cost(obj,x,z,u,params)
        %cost This method computes the cost, the Gauss-Newton step, the
        % first and (approximate) second derivatives of the cost with
        % respect to the step size, alpha, the approximate estimation error
        % covariance matrix (assuming that the optimal solution has been
        % reached for the nonlinear least-squares problem), and the first
        % derivative of the cost with respect to x.
        %
        % Inputs:
        %   x       The value of x for which we want to compute J
        %   z       The N x nz time history of measurements
        %   u       The N x nu time histopry of inputs
        %   params  A struct of constant model parameters
        %  
        % Outputs:
        %
        %   J           The nonlinear least-squares cost. Note that this is
        %               divided by 2 compared to the cost given in
        %               Bar-Shalom.
        %
        %   dx          The Gauss-Newton perturbation to xguess that is
        %               supposed to yield the optimal x.
        %
        %   dJdalpha    = dJ(xguess+alpha*delxgn)/dalpha evaluated at
        %               alpha = 0.
        %
        %   d2Jdalpha2  = d2J(xguess+alpha*delxgn)/dalpha2 evaluated at
        %               alpha = 0, except that the Hessian matrix d2Jdx2 
        %               used in this calculation is only the Gauss-Newton 
        %               approximation of the cost function Hessian.
        %
        %   P           The Gauss-Newton approximation of the estimation
        %               error covariance. It is the inverse of the approximate
        %               Hessian of J.
        %
        %   dJdx       The first partial derivative of J with respect to x.
        %
        
        % Dimensions
        N = size(z,1);
        nz = size(z,2);
        nx = size(x,1);
        
        % Efficiently compute the cost and return if applicable. Loop
        % through the measurements and set up the error vector and the
        % stacked measurement Jacobian matrix. Convert to ordinary
        % nonlinear least squares form using the Cholesky factor.
        if nargout == 1
            dza = zeros(nz*N,1);
            sidx = 1:nz;
            for k = 1:N
                uk = u(k,:).';
                zk = z(k,:).';
                % TODO: if R is an explicit function of k in the IEKF, this wont work...wrong k
                Rainvk = obj.getInvCholR(k); 
                hk = obj.MeasurementModel(k,x,uk,params);
                dza(sidx,1) = Rainvk'*(zk - hk);
                sidx = sidx + nz;
            end
            J = 0.5*(dza'*dza);
            return
        end

        % If we have arrived here, the search direction is needed. Compute
        % the cost as well as the array of the measurement model Jacobians.
        dza = zeros(nz*N,1);
        Ha = zeros(nz*N,nx);
        sidx = 1:nz;
        if isempty(u)
            u = zeros(N,0);
        end
        for k = 1:N
            uk = u(k,:).';
            zk = z(k,:).';
            Rainvk = obj.getInvCholR(k);
            [hk,Hk] = obj.MeasurementModel(k,x,uk,params);
            if size(Hk,2) < nx
                Hk = repmat(Hk,1,nx);
            end
            Ha(sidx,:) = Rainvk'*Hk;
            dza(sidx,1) = Rainvk'*(zk - hk);
            sidx = sidx + nz;
        end
        J = 0.5*(dza'*dza);

        % Use QR factorization to compute the Gauss-Newton search direction
        [Qb,Rb] = qr(Ha,0);
        delzb1 = Qb'*dza;
        dx = Rb\delzb1;
        Rbinv = inv(Rb);
        P = Rbinv*(Rbinv');
        dJdx = Ha'*dza;
        dJdalpha = dJdx'*dx;
        d2Jdalpha2 = (Rb*dx)'*(Rb*dx);
    end % cost

end % public methods

methods (Access=private)
    function Rainv = getInvCholR(obj,k)
        %Rainv Get the inverse of the cholsesly factor of the measurement
        % noise covariance at the current sample number.
        if ~isempty(obj.invCholR)
            Rainv = obj.invCholR;
            return
        end
        if isa(obj.MeasurementNoiseCovariance,'function_handle')
            Rainv = inv(chol(obj.MeasurementNoiseCovariance(k)));
        elseif size(obj.MeasurementNoiseCovariance,3) > 1
            Rainv = inv(chol(obj.MeasurementNoiseCovariance(:,:,k+1)));
        else
            Rainv = inv(chol(obj.MeasurementNoiseCovariance));
        end
    end % getInvCholR
end % private methods

end % classdef