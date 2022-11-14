function [hj,Hj,d2hjdx2] = measmodel_temp(tj,x,i1stdrv,i2nddrv)
%
%  Copyright (c) 2022 Jeremy W. Hopwood.  All rights reserved.
% 
%  This function gives the measurement function h(tj,x) and its first and
%  second derivatives with respect to x, Hj = dhj/dx and d2hjdx2 = dHj/dx.
%  It is for use in either a nonlinear least-squares problem or an extended
%  Kalman filter.
%
%  Inputs:
%
%   x           The vector of initial conditions at time 0.
%
%   tj          The time in seconds of the measurement.
%
%   i1stdrv     A character array/string or boolean that that defines
%               whether or not and how Hj is to be computed. It follows the
%               following logic: If i1stdrv=false or i1stdrv=0, neither the
%               first nor second partial derivatves of hj are computed. If
%               i1stdrv=true or i1stdrv=1, the default method is 'Analytic'
%               in which this function defines the analytic formula for Hj.
%               Alternatively, i1stdrv may be 'Analytic', 'CSDA', 'CFDM',
%               or 'FiniteDifference' to explicitly specify the computation
%               method.
%
%   i2nddrv     A character array/string or boolean that that defines
%               whether or not and how d2hjdx2 is to be computed. It
%               follows the same logic as i1stdrv.
%  
%  Outputs:
%
%   hj          The nz x 1 output vector.
%
%   Hj          = dhj/dx. Hj is a pxn matrix. This output will be needed to
%               do Newton's method or to do the Gauss-Newton method.
%
%   d2hjdx2     = d2hj/dx2. d2hjdx2 is a pxnxn array. This output will be
%               needed to do Newton's method.
%

% Define the dimension of hj and get the dimensions of x.
nz = 1;
nx = size(x,1);

% Set up output arrays as needed.
hj = zeros(nz,1);
if ~all(i1stdrv == 0)
    Hj = zeros(nz,nx);
    if ~all(i2nddrv == 0)
        d2hjdx2 = zeros(nz,nx,nx);
    else
        d2hjdx2 = [];
    end
else
    Hj = [];
    d2hjdx2 = [];
end

% Compute the hj outputs.
hj(1,1) = x(1);

% Return if neither first derivatives nor second derivatives need to be
% calculated.
if i1stdrv == 0
    return
end

% Calculate the first derivatives.
epsilon = sqrt(eps);
epsilon_inv = 1/epsilon;
%
% Analytic formula.
%
if all(i1stdrv==1) || strcmp(i1stdrv,'Analytic')
    Hj(1,1) = 1;
    Hj(1,2) = 0;
%
% Single Step Finite Difference.
%
elseif strcmp(i1stdrv,'FiniteDifference')
    for ii = 1:nx
        xplus = x;
        xplus(ii) =  x(ii) + epsilon;
        Hj(:,ii) = (measmodel_temp(tj,xplus,0,0)-hj)*epsilon_inv;
    end
%
% Complex-Step Derivative Approximation (CSDA)
%
elseif strcmp(i1stdrv,'CSDA')
    for ii = 1:nx
        xplus = x;
        xplus(ii) =  x(ii) + 1i*epsilon;
        Hj(:,ii) = imag(measmodel_temp(tj,xplus,0,0)).*epsilon_inv;
    end
%
% Central Finite Difference Method (CFDM)
%
elseif strcmp(i1stdrv,'CFDM')
    for ii = 1:nx
        xplus = x;
        xminus = x;
        xplus(ii) =  x(ii) + epsilon;
        xminus(ii) =  x(ii) - epsilon;
        Hj(:, ii) = 0.5*epsilon_inv*(measmodel_temp(tj,xplus,0,0) ...
            -measmodel_temp(tj,xminus,0,0));
    end
end

% Return if second derivatives do not need to be calculated.
if i2nddrv == 0
    return
end

% Calculate the second derivatives.
%
% Analytic formula.
%
if all(i2nddrv==1) || strcmp(i2nddrv,'Analytic')
    %
    % d2hjdx2 = 0 (already defined above). Normally this is quite a
    % complicated formula.
    return
end
%
% Loop through the columns of Hj and numerically compute their Jacobians.
%
% TODO: Double check the permutation of the numerical second derivatives.
%       Also check the accuracy of this approach.
%       Is there a better approach?
%
for jj = 1:nx
    %
    % Single Step Finite Difference.
    %
    if strcmp(i2nddrv,'FiniteDifference')
        for ii = 1:nx
            xplus = x;
            xplus(ii) =  x(ii) + epsilon;
            [~,Hjplus] = measmodel_temp(tj,xplus,i1stdrv,0);
            Hjpluscoljj = Hjplus(:,jj);
            Hjcoljj = Hj(:,jj);
            d2hjdx2(:,jj,ii) = (Hjpluscoljj-Hjcoljj)*epsilon_inv;
        end
    %
    % Complex-Step Derivative Approximation (CSDA)
    %
    elseif strcmp(i2nddrv,'CSDA')
        for ii = 1:nx
            xplus = x;
            xplus(ii) =  x(ii) + 1i*epsilon;
            [~,Hjplus] = measmodel_temp(tj,xplus,i1stdrv,0);
            Hjpluscoljj = Hjplus(:,jj);
            d2hjdx2(:,jj,ii) = imag(Hjpluscoljj).*epsilon_inv;
        end
    %
    % Central Finite Difference Method (CFDM)
    %
    elseif strcmp(i2nddrv,'CFDM')
        for ii = 1:nx
            xplus = x;
            xminus = x;
            xplus(ii) =  x(ii) + epsilon;
            xminus(ii) =  x(ii) - epsilon;
            [~,Hjplus] = measmodel_temp(tj,xplus,i1stdrv,0);
            [~,Hjminus] = measmodel_temp(tj,xminus,i1stdrv,0);
            Hjpluscoljj = Hjplus(:,jj);
            Hjminuscoljj = Hjminus(:,jj);
            d2hjdx2(:,jj,ii) = 0.5*epsilon_inv*(Hjpluscoljj - Hjminuscoljj);
        end
    end
end

end