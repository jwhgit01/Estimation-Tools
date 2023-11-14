function out = filterError(dt,z,u,xhat0,theta0,S0,Ptheta0,thetaIdx,ssfun,constants,guard,feParams,dispflag)
%filterError
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs parameter estimation using the linear filter error
% method on the system
%
%                dx/dt = A(th)*x + B(th)*u + F(th)*v                    (1)
%                    y = C(th)*x + D(th)*u                              (2)
%                 z(k) = y(tk) + G(th)*w(k)                             (3)            
%
% where v(t) is continuous-time, zero-mean, unit power spectral density white
% noise and w(k) is discrete-time, zero-mean, unit variance noise. This function
% implements the algorithm detailed in Chapter 5: Filter Error Method of "Flight
% Vehicle System Identification - A Time Domain Methodology" by Ravindra V.
% Jategaonkar. Equation numbers referencing this book are in the form Eq.
% (J.5.~). Chapter 6: Maximum Likelihood Methods of "Aircraft System
% Identification: Theory and Practice", 2nd edition by Eugene A. Morelli and
% Vladislav Klein is also used. Equations from this book are referenced in the
% form Eq. (M.6.~).
%
% Inputs:
%
%   dt      The sample time of the measurement vector, z, and input history, u.
%
%   z       The N x nz time history of measurements.
%
%   u       The N x nu time history of system inputs (optional). The first input
%           occurs at t = t0. If not applicable, set to an empty array, [].
%
%   xhat0   The nx x 1 initial state estimate.
%
%   theta0  The np x 1 initial parameter estimates.
%
%   S0      An initial guess of the model output residual covariance. Leave
%           empty, [], to approximate it.
%
%   Ptheta0 The covariance matrix of the initial guess, theta0. Leave empty, [],
%           if it is unknown.
%
%   thetaIdx  A struct with the fields A,B,C,D,F,G,bx,by each with the size
%           of their respective state equation matrices. The elements of these
%           matrices are equal to the parameter number or zero if that element
%           is known.
% 
%   ssfun   The function handle that returns the A,B,C,D,F,G matrices of the
%           LTI system in Eqs. (1)-(3), the jacobians of these matrices, as well
%           as the state equation and output equation bias vectors. The first
%           line of ssfun must be in the form
%               [sys,jac,bias] = ssfun(theta,parameterIdx,constants);
%
%   constants   A struct of constants that get passed to ssfun.
%
%   guard   A string defining which method to use in order to guard the Fisher
%           information matrix from being singular. Options are 'Constrained', 
%           'RankDeficient', 'LevenbergMarquardt', or 'PriorInformation'. If 
%           'PriorInformation' is chosen, Ptheta0 cannot be empty.
%
%   feParams A struct containing the following fields:
%               RelativeTolerance:      Convergence limit in terms of relative
%                                       change in the cost function
%               MaxIterations:          The maximum number of iterations
%               FCorrectionIteration:   Iteration number from which the estimate
%                                       of F will be corrected
%
%   dispflag A logical indicating whether to display the optimization
%            iterations.
%  
% Outputs:
%
%   out     The results struct containing the following fields:
%               ParameterEstimates
%               ParameterCovariance
%               OutputResidualCovariance
%               OutputPropogation
%               OutputEstimate
%               StatePropogation
%               StateEstimate
%               SteadyStateEstimateCovariance
%               SteadyStateKalmanGain
%               ParameterEstimateHistory
%               OutputResidualCovarianceHistory
%               FisherInformationMatrixHistory
%               ExitCondition
%               NumberOfIterations
% 

% Initialize algorithm variables
iter = 0; % Iteration number
khalf = 0; % Number of times step size has been halved
prevcost = Inf; % Cost funtion value from pervious iteration
updateF = false; % Logical whether to update the noise diffusion matrix
Fcompensation = false; % Logical whether to use F-compensation
theta = theta0; % current estimate of theta

% Get dimensions
N = size(z,1);
ny = size(z,2);
nth = size(theta0,1);

% Initial the linear system matrices given the initial parameter vector.
[sys,jac,bias] = ssfun(theta0,thetaIdx,constants);

% Initial output residual covariance matrix inverse
if isempty(S0)
    [sys,~,bias] = ssfun(theta0,thetaIdx,constants);
    [currentcost,Sinv,xhat,xbar,ytilde,sys.Phi,sys.Psi] = linearFilterErrorCost(sys,bias,xhat0,z,u,dt,[]);
    if dispflag
        disp(['cost function: det(R) - default value = ' num2str(currentcost)]);
    end
else
    Sinv = inv(S0);
end

% Initialize the result arrays
thetaHist = zeros(nth,feParams.MaxIterations);
thetaHist(:,1) = theta0;
fisherHist = zeros(nth,nth,feParams.MaxIterations);
if isempty(Ptheta0)
    fisherHist(:,:,1) = NaN(nth,nth);
else
    fisherHist(:,:,1) = Ptheta0;
end
SinvHist = zeros(ny,ny,feParams.MaxIterations);
SinvHist(:,:,1) = Sinv;
costFuntionHist = zeros(1,feParams.MaxIterations);
costFuntionHist(1) = currentcost;

% Iteration Loop
while iter <= feParams.MaxIterations

    while ~Fcompensation
        
        % Get the linear system matrices given the initial parameter vector.
        [sys,jac,bias] = ssfun(theta,thetaIdx,constants);

        % Compute the steady state Kalman gain, K.
        [K,Pxx] = linearFilterErrorKalmanGain(sys,Sinv,dt);

        % Compute the current cost
        [currentcost,SinvNew,xhat,xbar,ytilde,Phi,Psi] = linearFilterErrorCost(sys,bias,xhat0,z,u,dt,K);

        if ~updateF
            % Print out the iteration count, parameter values and cost function
            if dispflag
                disp(['iteration = ', num2str(iter)]);
                disp(['cost function: det(S) = ' num2str(currentcost)]);
            end
            Fcompensation = true;
            khalf = 0;
        else
            if dispflag
                disp('Correction of F (with the new estimated value of S)');
                disp(['cost function: det(S) = ' num2str(currentcost)]);
            end
            if currentcost > prevcost
                if dispflag
                    disp('Local divergence after F-compensation');
                end
                FIdx = thetaIdx.F;
                idx = FIdx > 0;
                khalf = khalf + 1;
                if khalf < 3 % Half the change in F
                    theta(FIdx(idx)) = (theta(FIdx(idx)) + FAlt(idx))/2;
                elseif khalf == 3 % Restore previous values
                    theta(FIdx(idx)) = FAlt(idx); 
                    Sinv = SinvAlt;
                else
                    Fcompensation = true;
                    khalf = 0;
                end
            else
                Fcompensation = true;
                khalf = 0;
            end
        end
        
    end

    % Store parameters and standard deviations only if successful step: convergence plot 
    if (iter > 0) && (~updateF || Fcompensation)
        thetaHist(:,iter+1) = theta;
        fisherHist(:,:,iter+1) = mathcalF;
        SinvHist(:,:,iter+1) = Sinv;
        costFuntionHist(iter+1) = currentcost;
    end

    % Check convergence
    [exitCondition,shouldHalf] = filterErrorConvergence(currentcost,prevcost,iter,khalf,feParams);
    if (exitCondition == 1 && ~Fcompensation) || (exitCondition > 1)
        break
    end
    if shouldHalf
        theta = (theta + thetaOld)/2;
        khalf  = khalf + 1;
        continue
    end
        
    % Compute gradients of the state estimate covariance matrix, Pxx
    dPdth = gradPLinear(sys,jac,dt,K,Pxx);

    % Compute gradients of Kalman gain matrix, K
    % TODO: Is it better to put this in gradP.m,?
    dKdth = pagemtimes(dPdth,sys.C'*Sinv) + pagemtimes(Pxx,pagemtimes(jac.C,'transpose',Sinv,'none'));

    % At each sample, compute the sensivity of the output.
    dybardth = gradybarLinear(sys,jac,xbar,xhat,u,ytilde,K,dKdth,Phi,Psi);

    % Also use forward finite difference to approximate sensitivities.
    % thetap = theta + 1e-6*abs(theta);
    % [sysp,jacp,biasp] = ssfun(thetap,ParameterIndex,Constants);
    % [~,~,xbar,~,~,~,~] = linearFilterErrorCost(sysp,biasp,dt,xhat0,Pxx0,u,z,estim);
    % TODO .... 

    % Sum over the samples to get cost function derivatives, mathcalF and
    % mathcalG
    mathcalF = zeros(nth,nth);
    mathcalG = zeros(nth,1);
    for k = 1:N
        mathcalF = mathcalF + (dybardth(:,:,k)'*Sinv)*dybardth(:,:,k); % Eq. (5.19)
        mathcalG = mathcalG + (dybardth(:,:,k)'*Sinv)*(ytilde(k,:).'); % Eq. (5.20)
    end

    % Compute parameter update using the Cholesky factor of mathcalF 
    cholF = chol(mathcalF);
    deltheta = cholF\(cholF'\mathcalG);

    % Guard against a singular Fisher information matrix
    deltheta = singularFisherGuard(sys,jac,K,dKdth,deltheta,mathcalF,guard);
     
    % Update parameter vector
    thetaOld = theta;
    theta = theta + deltheta;
    
    % Save the cost and iterate
    prevcost = currentcost;
    iter = iter + 1;
    Fcompensation = false;

    % Check whether we should use F-compensation using new S
    if iter < feParams.FCorrectionIteration 
        continue
    end

    % Compute system matrices using new theta
    [sys,jac,bias] = ssfun(theta,thetaIdx,constants);

    % Compute steady-state Kalman Gain matrix, K, and the a priori
    % steady state estimate error covariance, Pxx, using the old S.
    [K,Pxx] = linearFilterErrorKalmanGain(sys,Sinv,dt);

    % Compute the current cost
    [currentcost,SinvNew,xbar,xhat,ytilde,sys.Phi,sys.Psi] = linearFilterErrorCost(sys,bias,xhat0,z,u,dt,K);
    if dispflag
        disp(['iteration = ', num2str(iter)]);
        disp(['cost function: det(R) = ' num2str(currentcost)]);
    end

    % Check convergence
    [exitCondition,shouldHalf] = filterErrorConvergence(currentcost,prevcost,iter,khalf,feParams);
    if exitCondition == 1
        % On eps-convergence after the full iteration, no need to carry out
        % F-compensation, because at the true minimum it would not (or rather
        % should not) lead to any changes in det(R). Hence, terminate.
        thetaHist(:,iter+1) = theta;
        fisherHist(:,:,iter+1) = mathcalF;
        SinvHist(:,:,iter+1) = Sinv;
        break
    end
    if exitCondition == 2
        break
    end
    if shouldHalf
        theta = (theta + thetaOld)/2;
        khalf  = khalf + 1;
    else
        khalf = 0;
    end

    prevcost = currentcost;

    % Previous F and S value
    FAlt = sys.F;
    SinvAlt = Sinv;

    % Compensate estimate of mathcalF using new S
    FIdx = thetaIdx.F;
    theta = linearFisherCompensation(sys,FAlt,FIdx,SinvNew,Sinv,theta);
    Sinv = SinvNew;
    updateF = true;
               
end % end of main loop

% Compute the estimated outputs using the final model
ybar = zeros(N,ny);
yhat = zeros(N,ny);
for k = 1:N
    ybar(k,:) = (sys.C*xbar(k,:).' + sys.D*u(k,:).' + bias.y).';
    yhat(k,:) = (sys.C*xhat(k,:).' + sys.D*u(k,:).' + bias.y).';
end

% Remove zeros from history arrays
thetaHist(:,(iter+2):end) = [];
SinvHist(:,:,(iter+2):end) = [];
fisherHist(:,:,(iter+2):end) = [];

% Populate the output struct with the results.
out.ParameterEstimates = theta;
out.ParameterCovariance = inv(mathcalF);
out.OutputResidualCovariance = inv(Sinv);
out.OutputPropogation = ybar;
out.OutputEstimate = yhat;
out.StatePropogation = xbar;
out.StateEstimate = xhat;
out.SteadyStateEstimateCovariance = Pxx;
out.SteadyStateKalmanGain = K;
out.ParameterEstimateHistory = thetaHist;
out.OutputResidualCovarianceHistory = SinvHist;
out.FisherInformationMatrixHistory = fisherHist;
out.ExitCondition = exitCondition;
out.NumberOfIterations = iter;

end