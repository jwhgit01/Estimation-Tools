function [xhat,Jopt,P,termflag] = gaussNewtonEstimation(thist,zhist,uhist,R,h,xguess,derivmethod,dispflag,params)
%gaussNewtonEstimation
%
% Copyright (c) 2002 Mark L. Psiaki.     All rights reserved.
%               2022 Jeremy W. Hopwood.  All rights reserved.  
%
% This function does nonlinear batch least-squares estimation using
% the Gauss-Newton method. It aims to minimize the cost function
%
%              N  /              T     -1                \
%      J[x] = SUM | [z(i)-h(i,x)]   R(i)   [z(i)-h(i,x)] |
%             i=1 \                                      /
%
% Inputs:
% 
%   t           The N x 1 vector of measurement times or sample indices. If
%               t is given as an empty array, it is taken to be the
%               ordered sample indices of zhist.
%
%   z           The N x nx time history of measurements.
%
%   R           The measurement noise covariance. It may be given as a
%               constant nz x nz matrix, a nz x nz x N array, or a function
%               handle that is a function of thist and returns a nz x nz
%               matrix.
%
%   h           The name of the Matlab .m-file that contains the function
%               which defines the measurement model. It may be given as a
%               function handle, string, or char array. The function must
%               have the following form:
%
%                   [hj,Hj,~] = measmodel(tj,xj,i1stdrv,~,params)
%
%               where hj is the output evaluated at the current time step
%               or sample index tj and the state vector xj. The argument
%               i1stdrv is the method for computing the derivative of hj
%               with respect to xj, Hj. If it is equal to 0, the Jacobian
%               of hj is not computed and measmodel returns an empty array.
%               See 'derivmethod' below for more details.
%
%  xguess       The initial guess of x.
%
%  derivmethod  The method for computing the derivative of hj with
%               respect to xj, Hj. The default value, 1 = 'Analytic', is
%               the analytic formula for Hj. Other options may include
%               'FiniteDifference', 'CSDA', and 'CFDM'.
%
%  dispflag      A flag that tells whether (dispflag = 1) or not 
%                (dispflag = 0) to display interm optimization
%                results during the Gauss-Newton iterations.
%
%  params       A constant parameter struct, array, etc that is passed to
%               the measurement model, h.
%  
%  Outputs:
%
%  xhat         The final estimate of x.
%
%  Jopt         The final optimal value of the nonlinear least-squares
%               cost.  Note that this is divided by 2 compared to the  
%               cost given in Bar-Shalom.
%
%  P            The Gauss-Newton approximation of the estimation
%               error covariance.  It is the inverse of the approximate
%               Hessian of J.
%
%  termflag     A termination flag that tells whether the
%               procedure terminated well or not.  Its output values
%               have the following interpretations:
%
%                   0   Normal termination at the optimum
%
%                   1   Terminated because more than 50 step
%                       size halvings were required.  This
%                       may be an optimum.
%
%                   2   Terminated because more than 100
%                       Gauss-Newton iterations were required.
%                       This may not be the optimum.
%

% Calculate the initial cost and Gauss-Newton search direction.
[J,delxgn,dJdalpha,d2Jdalpha2,P,~] = gaussNewtonCost(xguess,thist,zhist,uhist,R,h,derivmethod,params);

% Predict the change in cost if a step size of alpha = 1 is taken.
delJpred = dJdalpha + 0.5*d2Jdalpha2;

% Decide whether to terminate now.
delJsizetest = abs(delJpred) < 1e-13*(1 + J);
delxsizetest = norm(delxgn) < 1e-09*(1 + norm(xguess));
termflag = 0;
if delJsizetest && delxsizetest
    xhat = xguess;
    Jopt = J;
    return
end

% Prepare quantities for use in controlling the Gauss-Newton iterations.
testdone = false;
niteration = 0;
iaflag = 0;

% Do one Gauss-Newton iteration per iteration of the following loop.
while ~testdone
    
    % Initial step size and Gauss-Newton iteration
    alpha = 1;
    xguessnew = xguess + alpha*delxgn;
    Jnew = gaussNewtonCost(xguessnew,thist,zhist,uhist,R,h,0,params);

    % Do step size halving if necessary in order to force a decrease
    % in the cost.
    nalphahalf = 0;
    while Jnew >= J
        nalphahalf = nalphahalf + 1;
        if nalphahalf > 50
            iaflag = 1;
            break
        end
        alpha = 0.5*alpha;
        xguessnew = xguess + alpha*delxgn;
        Jnew = gaussNewtonCost(xguessnew,thist,zhist,uhist,R,h,0,params);
    end
    
    % If we have halved the step size more than 50 times, make note.
    if iaflag == 1
        termflag = 1;
        break
    end

    % Update variables for next iteration.
    xguess = xguessnew;
    Jold = J;
    delJold = Jnew - J;
    delJpredold = delJpred;

    % Perform a Gauss-Newton iteration.
    [J,delxgn,dJdalpha,d2Jdalpha2,P,~] = gaussNewtonCost(xguess,thist,zhist,uhist,R,h,derivmethod,params);

    % Compute quantities in order to check whether we should terminate.
    delJpred = dJdalpha + .5*d2Jdalpha2;
    delJsizetest = abs(delJpred) < 1.e-13*(1 + J);
    delxsizetest = norm(delxgn) < 1.e-09*(1 + norm(xguess));
    alphatest = alpha == 1;
    delJratiotest = abs((delJold/delJpredold) - 1) < 0.01;
   
    % Criteria #1: The change in x and J is sufficiently small. 
    if delJsizetest && delxsizetest
        testdone = true;
    end

    % Criteria #2: The change in x is sufficiently small, the ratio of
    %              changes in delta J is sufficiently close to 1 for the
    %              case where the step size has not been halved.
    if alphatest && delJratiotest && delxsizetest
        testdone = true;
    end

    % If we have gotten to here, we have not converged nicely to a optimal
    % value. Perform more Gauss-Newton iterations.
    niteration = niteration + 1;

    % Criteria #3: We have succeeded the maximum number of Gauss-Newton
    %              iterations.
    if ~testdone && niteration >= 100
        termflag = 2;
        testdone = true;
    end

    % If desired, display the results of this Gauss-Newton iteration.
    if dispflag == 1
        fprintf('Iteration %02i : alpha = %06.4e, Jnew = %06.4e, Jold = %06.4e, norm(delxnew) = %06.4e\n',...
            niteration,alpha,J,Jold,norm(delxgn));
    end

end

% Populate necessary outputs with the results.
xhat = xguess;
Jopt = J;

end