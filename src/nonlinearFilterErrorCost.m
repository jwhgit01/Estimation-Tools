function [J,Rcal,yhat,xhat,Pxx,deltheta,Pcal] = nonlinearFilterErrorCost(...
                               t,z,u,f,h,Q,R,xhat0,Pxx0,theta,delthetaj,...
                                   nRK,constants,istepflag,filteringMethod)
%nonlinearFilterErrorCost
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function ...
%
% Inputs:
%
%   ...
%  
% Outputs:
%
%   ...
%

% Get the problem dimensions.
N = size(z,1);
nx = size(xhat0,1);
nz = size(z,2);
nv = size(Q,1);
np = size(theta,1);

% Anonymous functions that are in the required form for state estimation.
ffilt = @(t,x,u,vtil,dervflag,consts) f(t,x,u,vtil,dervflag,consts,theta);
hfilt = @(t,x,u,dervflag,consts) h(t,x,u,dervflag,consts,theta);

% Compute the estimated state time history using the current parameter
% estimates.
if strcmp(filteringMethod,'unscentedKalmanFilterDT')
    alpha = 0.01;
    beta = 2;
    kappa = 3-nx-nv;
    [xhat,Pxx] = unscentedKalmanFilterDT(t,z,u,ffilt,hfilt,Q,R,...
                                xhat0,Pxx0,nRK,alpha,beta,kappa,constants);
elseif strcmp(filteringMethod,'extendedKalmanFilterDT')
    [xhat,Pxx] = extendedKalmanFilterDT(t,z,u,ffilt,hfilt,Q,R,...
                                                 xhat0,Pxx0,nRK,constants);
else
    error('Unknown filtering method');
end

% TODO: If the state estimates have diverged, do something...
%
%
%
%

% Compute the model output using the current parameter estimates and 
% state estimates computed using the current parameter estimates. Also,
% estimate the model error covariance matrix, Rcal.
Rcal = zeros(nz,nz);
yhat = zeros(N,nz);
for k = 1:N
    tk = t(k+1,1);
    zk = z(k,:).';
    xhatk = xhat(k+1,:).';
    uk = u(k,:).';
    yhatk = h(tk,xhatk,uk,0,constants,theta);
    yhat(k,:) = yhatk.';
    nuk = zk - yhatk;
    Rcal = Rcal + nuk*nuk';
end
Rcal = Rcal/N;
% Another option is to take only the diagonal elements:
% Rcal = diag(diag(Rcal))/N;

% Compute the filter error cost function.
J = det(Rcal);

% If only the cost is needed, return.
if istepflag == 0
    deltheta = [];
    Pcal = [];
    return
end

% Compute the perturbed model output and numerically compute sensitivities.
dydtheta = zeros(nz,np,N);
for jj = 1:np
    ej = zeros(np,1);
    ej(jj) = 1;
    thetaplus = theta + delthetaj(jj)*ej;
    [~,~,yhatplus] = nonlinearFilterErrorCost(t,z,u,f,h,Q,R,xhat0,Pxx0,...
                      thetaplus,delthetaj,nRK,constants,0,filteringMethod);
    for k = 1:N
        yhatplusk = yhatplus(k,:).';
        yhatk = yhat(k,:).';
        dydtheta(:,jj,k) = (yhatplusk-yhatk)/(delthetaj(jj));
    end
end

% Compute Fcal and Gcal uaing Eqs. (5.19) and (5.20).
Fcal = zeros(np,np);
Gcal = zeros(np,1);
for k = 1:N
    dydthetak = dydtheta(:,:,k);
    zk = z(k,:).';
    yhatk = yhat(k,:).';
    nuk = zk - yhatk;
    Fcal = Fcal + (dydthetak'/Rcal)*dydthetak;
    Gcal = Gcal - (dydthetak'/Rcal)*nuk;
end

% Compute deltheta that satisfies Eq. (5.18) using Cholesky factorization.
cholFcal = chol(Fcal);
deltheta = -cholFcal\(cholFcal'\Gcal);

% Compute the parameter estimate covariance using the information matrix.
Pcal = inv(Fcal);

end