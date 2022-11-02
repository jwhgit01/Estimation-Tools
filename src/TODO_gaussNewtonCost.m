function [J,dx,dJdalpha,d2Jdalpha2,P,dJdx] = gaussNewtonCost(xguess,t,z,R,hj,dxflag)
%
%  Copyright (c) 2002 Mark L. Psiaki.    All rights reserved. 
%                2022 Jeremy W. Hopwood. All rights reserved.
% 
%  This function computes the cost, the Gauss-Newton step, the first and
%  (approximate) second derivatives of the cost with respect to the step
%  size, alpha, the approximate estimation error covariance matrix
%  (assuming that the optimal solution has been reached for the nonlinear
%  least-squares problem), and the first derivative of the cost with
%  respect to x.
%
%  Inputs:
%
%    xguess     The current guess of x.
%
%    t          The Nx1 vector of radar measurement times.
%
%    z          The kx2 time history of measurements.
%
%    R          The measurement noise covariance.
%
%    hj         The name of the Matlab .m-file that contains the
%               function which defines the measurement model h(t,x). This may
%               be a character string or a function handle.  For example,
%               if the measurement model is contained in the file
%               hjmissile.m with the function name hjmissile, then on input
%               to the present function hj must equal 'hjmissile' or
%               @hjmissile, and the first line of the file hjmissile.m must
%               be:
%
%                   function [hj,Hj,d2hjdx2] = hjmissile(tj,x,R)
%
%               The function must be written so that hj defines the output
%               of the measurement model as a function of x and t, and so
%               that Hj and d2hjdx2 and the first and second matrix partial
%               derivatives of hj with respect to x if dervflag = 1.  If
%               dervflag = 0, then these outputs must be empty arrays.
%
%    dxflag     A flag that tells whether (dxflag = 1) or not (dxflag = 1)
%               to compute the other outputs besides J.
%  
%  Outputs:
%
%    J          The nonlinear least-squares cost.  Note that this
%               is divided by 2 compared to the cost given in
%               Bar-Shalom.
%
%    dx         The Gauss-Newton perturbation to xguess that is
%               supposed to yield the optimal x.
%
%    dJdalpha   = dJ(xguess+alpha*delxgn)/dalpha evaluated at
%               alpha = 0.
%
%    d2Jdalpha2 = d2J(xguess+alpha*delxgn)/dalpha2 evaluated at
%               alpha = 0, except that the Hessian matrix d2Jdx2 
%               used in this calculation is only the Gauss-Newton 
%               approximation of the cost function Hessian.
%
%    P          The Gauss-Newton approximation of the estimation
%               error covariance.  It is the inverse of the approximate
%               Hessian of J.
%
%    dJdx       The first partial derivative of J with respect to x.
%

% Get the number and dimension of the measurements and the dimension of x.
N = size(t,1);
n = size(xguess,1);
p = size(z,1);


% Loop through the measurement times and set up the error vector and the
% large measurement Jacobian matrix.  Re-normalize so as to deal with a
% nonlinear measurement equation that has a measurement error with the
% identity matrix for its covariance.
delzabigvec = zeros(p*N,1);
if dxflag == 1
    Habigmat = zeros(p*N,n);
end
idumvec = (1:p).';
Rainv = inv(R);
for j = 1:k
    [hjmod,Hjmod] = hj(xguess,t(j,1),lradar,dxflag,0);
    hjmeas = z(j,:)';
    delzabigvec(idumvec,1) = Rainv*(hjmeas - hjmod);
    if dxflag == 1
        Habigmat(idumvec,:) = Rainv*Hjmod;
    end
    idumvec = idumvec + 2;
end
%
%  Compute the cost.
%
   J = 0.5*(delzabigvec'*delzabigvec);
%
%  Return if only the cost is needed.
%
   if dxflag == 0
      return
   end
%
%  Compute the Gauss-Newton search direction.  Use QR factorization.
%
   [Qb,Rb] = qr(Habigmat,0);
   delzb1bigvec = (Qb')*delzabigvec;
   Rbinv = inv(Rb);
   dx = Rbinv*delzb1bigvec;
   P = Rbinv*(Rbinv');
   dJdx = (Habigmat')*delzabigvec;
   dJdalpha = (dJdx')*dx;
   dum = Rb*dx;
   d2Jdalpha2 = (dum')*dum;