function dPdt = covarianceDynamicsCT(Pvec,A,Q)
%covarianceDynamicsCT 
%
%  Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
%  This function computes the linear, continuous-time dynamics of
%  covariance of the state estimate in a Kalman filter, extended Kalman
%  filter, etc.
%
%  Inputs:
%
%    PVec           The vector form of the covariance matrix, Pvec = P(:).
%
%    A              The jacobian of the process dynamics at the current
%                   state estimate and time.
%
%    Q              The continuous-time process noise covariance at the
%                   current time.
%  
%  Outputs:
%
%    dPdt           The vector form of the time derivative of the
%                   covariance of the state estimate.
% 

P = reshape(Pvec,size(A));

Pdot = A*P + P*A.' + Q;

dPdt = Pdot(:);

end