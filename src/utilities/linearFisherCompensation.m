function theta = linearFisherCompensation(sys,F_alt,F_idx,SinvNew,Sinv,theta)

% Revise the estimates of the process noise distribution matrix F to
% compensate for the RI revision by heuristic procedure (Eq. 5.41)
%
% Chapter 5: Filter Error Method
% "Flight Vehicle System Identification - A Time Domain Methodology"
% Author: Ravindra V. Jategaonkar
% Published by AIAA, Reston, VA 20191, USA
%
% Filter error method for linear systems (ml_fem_linear)
%
% Inputs:
%    NF            number of free parameters of process noise distribution matrix F
%    jF            indices of unknown parameters appearing in F-matrix
%    FAlt          elements of F matrix prior to compensation
%    RInew         inverse of updated R 
%    RIprev        inverse of R from the previous step
%    Cmat          linearized observation matrix
%    parVal        parameter vector
%
% Outputs:
%    parVal        parameter vector with compensated F-elements


% Get the "C" matrix
C = sys.C;

% If some or all of the F-elements (diagonal) are free, then update them:
if  any(find(F_idx))
    
    % Use cholesky factor to get the diagonal elements of inv(S), r.
    r = diag(Sinv);
    rnew = diag(SinvNew);

    % Correct the noise diffusion matric, F.
    correction = ((C.^2)'*(r.*sqrt(r./rnew)))./((C.^2)'*r);
    F = F_alt.*correction; % Eq. (J.5.41)

    % Update theta to reflect new F
    idx = F_idx > 0;
    theta(F_idx(idx)) = F(idx);
    
end

return
% end of function
