function Pkp1 = covarianceUpdateDT(Pk,F,Q)
%covarianceUpdateDT 
%
%  Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
%  This function computes the linear, discrete-time update covariance of
%  the state estimate in a Kalman filter, extended Kalman filter, etc.
%
%  Inputs:
%
%    Pk             The covariance of the state estimate at sample k.
%
%    F              The discrete-time jacobian of the process model at the
%                   current state estimate and time step.
%
%    Q              The discrete-time process noise covariance at the
%                   current time step.
%  
%  Outputs:
%
%    Pkp1           The covariance of the state estimate at sample k+1.
% 

TODO

end