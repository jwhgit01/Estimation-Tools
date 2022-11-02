function [xhat,Jopt,P,termflag] = gaussNewtonEstimation(t,z,R,xguess,dispflag)
%gaussNewtonEstimation
%
%  Copyright (c) 2002 Mark L. Psiaki.    All rights reserved.  
%                2022 Jeremy W. Hopwood. All rights reserved.
%
%  This function does nonlinear batch least-squares estimation using
%  the Gauss-Newton method.
%
%  Inputs:
%
%    t           The Nx1 vector of measurement times.
%
%    z           The Nxn time history of measurements.
%
%    R           The measurement noise covariance.
%
%    xguess      The initial guess of x.
%
%    dispflag    A flag that tells whether (dispflag = 1) or not 
%                (dispflag = 0) to display interm optimization
%                results during the Gauss-Newton iterations.
%  
%  Outputs:
%
%    xhat        The final estimate of x.
%
%    Jopt        The final optimal value of the nonlinear least-squares
%                cost.  Note that this is divided by 2 compared to the  
%                cost given in Bar-Shalom.
%
%    P           The Gauss-Newton approximation of the estimation
%                error covariance.  It is the inverse of the approximate
%                Hessian of J.
%
%    termflag    A termination flag that tells whether the
%                procedure terminated well or not.  Its output values
%                have the following interpretations:
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


% Get the number of measurements.
N = size(t,1);

% Calculate the initial cost and Gauss-Newton search direction.
[J,delxgn,dJdalpha,d2Jdalpha2,P,~] = jdxgn(xguess,t,rhoahist,rhobhist,la,lb,sigma,1);

% Predict the change in cost if a step size of alpha = 1 is taken.
delJpred = dJdalpha + 0.5*d2Jdalpha2;

% Decide whether to terminate now.
delJsizetest = abs(delJpred) < 1.e-13*(1 + J);
delxsizetest = norm(delxgn) < 1.e-09*(1 + norm(xguess));
termflag = 0;
if delJsizetest && delxsizetest
    xhat = xguess;
    Jopt = J;
    return
end

% Prepare quantities for use in controlling the Gauss-Newton iterations.
testdone = 0;
niteration = 0;
iaflag = 0;

% Do one Gauss-Newton iteration per iteration of the following loop.
while testdone == 0
    alpha = 1;
    xguessnew = xguess + alpha*delxgn;
    Jnew = jdxgncart(xguessnew,t,rhoahist,rhobhist,la,lb,sigma,0);

    % Do step size halving if necessary in order to force a decrease
    %  in the cost.
    nalphahalf = 0;
    while Jnew >= J
        nalphahalf = nalphahalf + 1;
        if nalphahalf > 50
            iaflag = 1;
            break
        end
        alpha = 0.5*alpha;
        xguessnew = xguess + alpha*delxgn;
        Jnew = jdxgncart(xguessnew,t,rhoahist,rhobhist,la,lb,sigma,0);
    end
    if iaflag == 1
        termflag = 1;
        break
    end
    xguess = xguessnew;
    Jold = J;
    delJold = Jnew - J;
    delJpredold = delJpred;
    [J,delxgn,dJdalpha,d2Jdalpha2,P,~] = jdxgncart(xguess,t,rhoahist,rhobhist,la,lb,sigma,1);     
    delJpred = dJdalpha + .5*d2Jdalpha2;
    delJsizetest = abs(delJpred) < 1.e-13*(1 + J);
    delxsizetest = norm(delxgn) < 1.e-09*(1 + norm(xguess));
    alphatest = alpha == 1;
    delJratiotest = abs((delJold/delJpredold) - 1) < 0.01;
    if delJsizetest && delxsizetest
        testdone = 1;
    end
    if alphatest && delJratiotest && delxsizetest
        testdone = 1;
    end
    niteration = niteration + 1;                
    if testdone == 0 && niteration >= 100
        termflag = 2;
        testdone = 1;
    end
    if dispflag == 1
        disp([' At iteration ',int2str(niteration),' alpha = ',...
            num2str(alpha),', Jnew = ',num2str(J),', Jold = ',...
            num2str(Jold),', and norm(delxnew) = ',...
            num2str(norm(delxgn)),'.'])
    end
end

xhat = xguess;
Jopt = J;

end