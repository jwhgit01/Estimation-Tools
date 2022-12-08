function [J,dx,dJdalpha,d2Jdalpha2,P,dJdx] = gaussNewtonCost(xguess,thist,zhist,uhist,R,h,idxflag,params)
%
% Copyright (c) 2002 Mark L. Psiaki.     All rights reserved. 
%               2022 Jeremy W. Hopwood.  All rights reserved.
% 
% This function computes the cost, the Gauss-Newton step, the first and
% (approximate) second derivatives of the cost with respect to the step
% size, alpha, the approximate estimation error covariance matrix
% (assuming that the optimal solution has been reached for the nonlinear
% least-squares problem), and the first derivative of the cost with
% respect to x.
%
% Inputs:
%
%   xguess      The current guess of x.
%
%   thist       The Nx1 vector of measurement times or sample indices. If
%               it is empty, take it to be an ordered index array the size
%               of zhist.
%
%   zhist       The N x nz time history of measurements.
%
%   R           The measurement noise covariance. If it is given as a 
%               nz x nz matrix, then it is assumed to be constant and not a
%               function of the sample time or index. Otherwise, it may be
%               given as a nz x nz x N array or a function handle that is a
%               function of t (or k depending on thist).
%
%   hj          The name of the Matlab .m-file that contains the
%               function which defines the measurement model h(t,x) or
%               h(k,x). This may be a function handle string or char array.
%               For example, if the measurement model is contained in the
%               file hjmissile.m with the function name hjmissile, then on
%               input to the present function hj must equal 'hjmissile' or
%               @hjmissile, and the first line of the file hjmissile.m must
%               have the form:
%
%                   function [hj,Hj,~] = hjmissile(t,x,i1stdrv,~,params)
%
%               where hj is the output evaluated at the current time step
%               or sample index tj and the state vector xj. The argument
%               i1stdrv is the method for computing the derivative of hj
%               with respect to xj, Hj. The default value, 1 = 'Analytic',
%               is the analytic formula for Hj. Other options may include
%               'FiniteDifference', 'CSDA', and 'CFDM'. If it is equal to
%               0, the Jacobian of hj is not computed and measmodel returns
%               an empty array.
%
%   idxflag     A flag that tells whether (idxflag = 1) or not
%               (idxflag = 1) to compute the other outputs besides J.
%  
% Outputs:
%
%   J           The nonlinear least-squares cost. Note that this
%               is divided by 2 compared to the cost given in
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

% Get the number and dimension of the measurements and the dimension of x.
N = size(zhist,1);
nz = size(zhist,2);
nx = size(xguess,1);

% If thist is empty, make it the sample indices
if isempty(thist)
    thist = (1:N).';
end

% Pre-compute the inverse of the cholesky decompisition of R if necessary
if ~isa(R,'function_handle') && size(R,3) == 1
    [cholR,flag] = chol(R);
    if flag>0
        pause
    end
    invcholR = inv(cholR);
end

% Loop through the measurements and set up the error vector and the
% stacked measurement Jacobian matrix.  Re-normalize so as to deal with a
% nonlinear measurement equation that has a measurement error with the
% identity matrix for its covariance.
delza = zeros(nz*N,1);
if ~all(idxflag == 0)
    Ha = zeros(nz*N,nx);
end
sidx = 1:nz;
for k = 1:N
    uj = uhist(k,:).';
    [hj,Hj] = feval(h,thist(k,1),xguess,uj,idxflag,params);
    zj = zhist(k,:).';
    if ~isa(R,'function_handle')
        if size(R,3) > 1
            Rainvk = inv(chol(R(:,:,k)));
        else
            Rainvk = invcholR;
        end
    else
        Rainvk = inv(chol(R(thist(k,1))));
    end
    delza(sidx,1) = Rainvk'*(zj - hj);
    if ~all(idxflag == 0)
        Ha(sidx,:) = Rainvk'*Hj;
    end
    sidx = sidx + nz;
end

% Compute the cost.
J = 0.5*(delza'*delza);

% Return if only the cost is needed.
if all(idxflag == 0)
    return
end

% Compute the Gauss-Newton search direction. Use QR factorization.
[Qb,Rb] = qr(Ha,0);
delzb1 = Qb'*delza;
Rbinv = inv(Rb);
dx = Rbinv*delzb1;
P = Rbinv*(Rbinv');
dJdx = Ha'*delza;
dJdalpha = dJdx'*dx;
dum = Rb*dx;
d2Jdalpha2 = dum'*dum;

end