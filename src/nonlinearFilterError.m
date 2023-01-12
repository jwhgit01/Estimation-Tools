function out = nonlinearFilterError(t,z,u,f,h,Q,R,xhat0,Pxx0,theta0,...
                          delthetaj,nRK,constants,filteringMethod,dispflag)
%nonlinearFilterError
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs parameter estimation using the nonlinear filter
% error method on the system
%
%                     dxdt = f(t,x,u,v;theta)                       (1)
%                     y(t) = h(t,x,u;theta)                         (2)
%                     z(k) = h(tk,x(tk),u(tk);theta) + w(k)         (3)            
%
% where v(t) is continuous-time, zero-mean, white noise with constant power
% spectral density Q and w(k) is discrete-time, zero-mean, white noise with
% constant covariance R. This function implements the algorithm detailed in
% Chapter 5: Filter Error Method of "Flight Vehicle System Identification -
% A Time Domain Methodology" by Ravindra V. Jategaonkar, Published by AIAA.
% Equation numbers referenced in this function refer to this book.
%
% Inputs:
%
%   t       The N x 1 sample time vector. If f is a discrete-time dynamic
%           model, t must be givenn as an empty array, []. The first sample
%           occurs after the initial condition of t = t0 or k = 0;
%
%   z       The N x nz time history of measurements.
%
%   u       The N x nu time history of system inputs (optional). The first
%           input occurs at t = t0 or k = 0. If not applicable set to an
%           empty array, [].
% 
%   f       The function handle that computes the continuous-time dynamics
%           given a parameter vector, p. The first line of f must
%           be in the form
%               [f,A,D] = nonlindyn(t,x,u,vtil,dervflag,constants,p)
% 
%   h       The function handle that computes the modeled output of the
%           system. The first line of h must be in the form
%               [h,H] = measmodel(t,x,u,dervflag,constants,p)
%   
%   Q,R     The discrete-time process and measurement noise covariance.
%           These may be specificed as constant matrices, ()x()xN
%           3-dimensional arrays, or function handles that are functions of
%           the sample number, k.
%
%   xhat0   The nx x 1 initial state estimate.
%
%   P0      The nx x nx symmetric positive definite initial state
%           estimation error covariance matrix.
%
%   nRK     The number of intermediate Runge-Kutta integrations steps used
%           in the discretization of the nonlinear dynamics. May be given
%           as an empty array to use the default number.
%
%   params  A struct of constants that get passed to the dynamics model and
%           measurement model functions.
%
%   nonlinearFilter The filtering/smoothing method used ....
%  
% Outputs:
%
%   ...
% 

% Compute the initial cost and Gauss-Newton search direction.
disp('Computing initial cost and sensitivities...');
thetahat = theta0;
[J,Rcal,yhat,xhat,Pxx,deltheta,Pcal] = nonlinearFilterErrorCost(...
                                t,z,u,f,h,Q,R,xhat0,Pxx0,thetahat,...
                                delthetaj,nRK,constants,1,filteringMethod);

% Initialize output arrays
thetahist = theta0;
thetavarhist = diag(Pcal);

% Decide whether to terminate now based on the size of deltheta and Pcal.
standarderrortest = norm(diag(Pcal)./abs(thetahat)) < 1e-09;
delthetatest = norm(deltheta) < 1e-09*(1 + norm(thetahat));
if delthetatest && standarderrortest
    out.ParameterEstimates = thetahat;
    out.ParameterCovariance = Pcal;
    out.ModelErrorCovariance = Rcal;
    out.EstimatedOutput = yhat;
    out.StateEstimate = xhat;
    out.StateEstimateCovariance = Pxx;
    out.ParameterEstimateHistory = thetahist;
    out.ParameterVarianceHistory = thetavarhist;
    out.TerminationFlag = 0;
    return
end

% Prepare quantities for use in controlling the Gauss-Newton iterations.
shouldExit = false;
niteration = 0;
maxiteration = 50;
iaflag = 0;
tolRcal = 1e-4;
termflag = 0;

% Iterate until the convergence criteria is met or a maximum number of
% iterations is met.
while ~shouldExit

    % Initial step size and Gauss-Newton iteration.
    disp('Computing cost using new theta...');
    alpha = 1;
    thetahatnew = thetahat + alpha*deltheta;
    Jnew = nonlinearFilterErrorCost(t,z,u,f,h,Q,R,xhat0,Pxx0,...
                    thetahatnew,delthetaj,nRK,constants,0,filteringMethod);

    % Do step size halving if necessary in order to force a decrease
    % in the cost.
    nalphahalf = 0;
    while Jnew >= J
        disp('Halving step size...');
        nalphahalf = nalphahalf + 1;
        if nalphahalf > 50
            iaflag = 1;
            break
        end
        alpha = 0.5*alpha;
        thetahatnew = thetahat + alpha*deltheta;
        disp('Computing cost using new halved step size...');
        Jnew = nonlinearFilterErrorCost(t,z,u,f,h,Q,R,xhat0,Pxx0,...
                    thetahatnew,delthetaj,nRK,constants,0,filteringMethod);
    end

    % If we have halved the step size more than 50 times, make note.
    if iaflag == 1
        termflag = 1;
        break
    end

    % Update variables for next iteration.
    thetahat = thetahatnew;
    Jold = J;
    delJold = Jnew - J;

    % Perform a Gauss-Newton iteration.
    disp('Computing cost and sensitivities using new theta...');
    [J,Rcal,yhat,xhat,Pxx,deltheta,Pcal] = nonlinearFilterErrorCost(...
                                t,z,u,f,h,Q,R,xhat0,Pxx0,thetahat,...
                                delthetaj,nRK,constants,1,filteringMethod);
    
    % Store the results.
    thetahist = [thetahist thetahat];
    thetavarhist = [thetavarhist diag(Pcal)];

    % Compute quantities in order to check whether we should terminate.
    delJtest = abs(delJold/J) > tolRcal;
    delthetatest = norm(deltheta) < 1e-09*(1 + norm(thetahat));
   
    % Convergence Criteria: The change in theta and J is sufficiently small
    if delJtest && delthetatest
        testdone = true;
    end

    % If we have gotten to here, we have not converged nicely to a optimal
    % value. Perform more Gauss-Newton iterations.
    niteration = niteration + 1;

    % Stop is we have succeeded the maximum number of iterations.
    if ~testdone && niteration >= maxiteration
        termflag = 2;
        testdone = true;
    end

    % If desired, display the results of this Gauss-Newton iteration.
    if dispflag == 1
        fprintf('Iteration %02i : alpha = %06.4e, Jnew = %06.4e, Jold = %06.4e, norm(deltheta) = %06.4e\n',...
            niteration,alpha,J,Jold,norm(deltheta));
    end

end % while loop

% Populate the output struct with the results.
out.ParameterEstimates = thetahat;
out.ParameterCovariance = Pcal;
out.ModelErrorCovariance = Rcal;
out.EstimatedOutput = yhat;
out.StateEstimate = xhat;
out.StateEstimateCovariance = Pxx;
out.ParameterEstimateHistory = thetahist;
out.ParameterVarianceHistory = thetavarhist;
out.TerminationFlag = termflag;

end