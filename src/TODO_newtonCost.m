function [J,dx,dJdalpha,d2Jdalpha2,P,dJdx,d2Jdx2,flagHess] = newtonCost(xguess,t,z,R,dxflag)
%newtonCost
%
%  Copyright (c) 2002 Mark L. Psiaki.    All rights reserved. 
%                2022 Jeremy W. Hopwood. All rights reserved.
% 
%  This function computes the cost, the Newton step, the first and second
%  derivatives of the cost with respect to the step size, alpha, the
%  approximate estimation error covariance matrix (assuming that the
%  optimal solution has been reached for the nonlinear least-squares
%  problem), and the first and second derivatives of the cost with respect
%  to x.
%
%  Inputs:
%
%    xguess      The current guess of x.
%
%    t           The Nx1 vector of radar measurement times.
%
%    z           The kx2 time history of measurements.
%
%    R           The measurement noise covariance.
%
%    dxflag      A flag that tells whether (dxflag = 1) or not (dxflag = 1)
%                to compute the other outputs besides J.
%  
%  Outputs:
%
%    J           The nonlinear least-squares cost.  Note that this
%                is divided by 2 compared to the cost given in
%                Bar-Shalom.
%
%    dx          The Newton perturbation to xguess that is
%                supposed to yield the optimal x.
%
%    dJdalpha    = dJ(xguess+alpha*delxgn)/dalpha evaluated at
%                alpha = 0.
%
%    d2Jdalpha2  = d2J(xguess+alpha*delxgn)/dalpha2 evaluated at
%                alpha = 0.
%
%    P           The 4x4 Newton approximation of the estimation
%                error covariance.  It is the inverse of the approximate
%                Hessian of J.  This will be the inverse of the
%                exact d2Jdx2 Hessian matrix if it is positive 
%                definite.
%
%    dJdx        The first partial derivative of J with respect to x.
%
%    d2Jdx2      The second partial derivative of J with respect to x.
%
%    flagHess   A flag that tells whether (flagHess = 1) or not
%                (flagHess = 0) the Hessian of the cost function
%                had to get modified by adding a positive
%                definite matrix to it in order to get it to be
%                positive definite so that the Newton calculation
%                would produce a direction in which a cost decrease
%                could be guaranteed to occur.

% Get the number of measurements.
N = size(t,1);

% Loop through the measurement times and set up the error vector
%  and the large measurement Jacobian matrix.  Re-normalize
%  so as to deal with a nonlinear measurement equation that has
%  a measurement error with the identity matrix for its covariance.
%
   delzabigvec = zeros(2*k,1);
   if dxflag == 1
      Habigmat = zeros(2*k,4);
   end
   idumvec = [1:2]';
   Rainv = diag([(1/sigmarho);(1/sigmatheta)]);
   V2ndO = zeros(4,4);
   for j = 1:k
      [hjmod,Hjmod,d2hjdx2] = hjmissle(xguess,t(j,1),lradar,dxflag,1);
      hjmeas = z(j,:)';
      delzabigvec(idumvec,1) = Rainv*(hjmeas - hjmod);
      if dxflag == 1
         Habigmat(idumvec,:) = Rainv*Hjmod;
         dum = d2hjdx2(1,:,:);
         dum1 = zeros(4,4);
         dum1(:) = dum(:);
         dum1 = dum1*(Rainv(1,1)*delzabigvec(1,1));
         V2ndO = V2ndO - dum1;
         dum = d2hjdx2(2,:,:);
         dum1(:) = dum(:);
         dum1 = dum1*(Rainv(2,2)*delzabigvec(2,1));
         V2ndO = V2ndO - dum1;
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
%  Compute the Newton search direction.  Use QR factorization.
%
   [Qb,Rb] = qr(Habigmat,0);
   delzb1bigvec = (Qb')*delzabigvec;
   Rbinv = inv(Rb);
%
%  The following calculations incorporate the 2nd derivative
%  terms of h(x) into the Hessian.  They do this in a way that
%  preserves the use of square-root calculations.  Also, if 
%  the resulting cost Hessian is not positive definite, 
%  then a positive definite increment gets added for
%  purposes of computing the Newton search direction so that
%  it can be guaranteed to be a descent direction of the cost
%  function.
%
   Dmat = eye(4) + (Rbinv')*(V2ndO*Rbinv);
%
%  Compute the Newton search direction.  Use QR factorization.
%
   evals = eig(Dmat);
   elim = 1.e-12*max(abs(evals));
   emin = min(evals);
   flagHess = 0;
   if emin < elim
      Dmat = Dmat + eye(4)*(elim - emin);
      flagHess = 1;
   end
   Rmat = chol(Dmat);
   Rmat_inv = inv(Rmat);
   Rtotinv = Rbinv*Rmat_inv;
   dx = Rtotinv*((Rmat_inv')*delzb1bigvec);
   P = Rtotinv*(Rtotinv');
   dJdx = (Habigmat')*delzabigvec;
   d2Jdx2 = (Rb')*Rb + V2ndO;
   dJdalpha = (dJdx')*dx;
   d2Jdalpha2 = (dx')*(d2Jdx2*dx);