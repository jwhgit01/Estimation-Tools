function deltheta = singularFisherGuard(sys,jac,K,dKdth,deltheta,mathcalF,guard)
%singularFisherGuard 

% Get necessary system matrices and their gradients
C = sys.C;
dCdth = jac.C;

% Get dimensions
nx = size(C,2);
nth = size(dCdth,3);

if strcmp(guard,'Constrained')
    
    % Compute gradient of KC-diagonal, Eq. (J.5.36)
    gradKCdiag = zeros(nx,nth);
    for pp = 1:nth
        gradKCdiag(:,pp) = diag(dKdth(:,:,pp)*C + K*dCdth(:,:,pp));
    end

    % Inverse of the second gradient of the cost function using chol
    cholF = chol(mathcalF);
    d2Jinv = cholF\(cholF'\eye(nth));

    % Linear approximation of the constraint, Eq. (J.5.35)
    diagKC = diag(K*C);
    KCineq = diagKC + gradKCdiag*deltheta - 1;

    % Check if any constraints are violated
    violations = KCineq > 0;
    ell = length(find(violations));
    if ell == 0
        return
    end
    
    % Constrined solution
    % s = ones(ell,1) - diagKC(violations); % Eq. (J.5.37)
    s = KCineq(violations); % From kcle1_lin.m ?
    M = gradKCdiag(violations,:); % Eq. (J.5.38)
    cholMd2JinvMtp = chol(M*d2Jinv*M');
    delthetaConstr = (d2Jinv*M')*(cholMd2JinvMtp\(cholMd2JinvMtp'\s));
    deltheta = deltheta - delthetaConstr; % Eq. (J.5.39)   

else
    warning('Invalid guard! Fisher information matrix may be singular.')
end

end