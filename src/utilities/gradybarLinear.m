function dybardth = gradybarLinear(sys,jac,xbar,xhat,u,ytilde,K,dKdth,Phi,Psi)

% Get necessary state equation matrices
C = sys.C;

% Get necessary Jacobians 
dAdth = jac.A;
dBdth = jac.B;
dCdth = jac.C;
dDdth = jac.D;

% Dimensions
N = size(ytilde,1);
ny = size(ytilde,2);
nx = size(xbar,2);
nth = size(dAdth,3);

% The Jacobians of the bias vectors
dbxdth = zeros(nx,nth);
dbxdth(1:nx,(nth-ny-nx+1):(nth-ny)) = eye(nx);
dbydth = zeros(ny,nth);
dbydth(1:ny,(nth-ny+1):nth) = eye(ny);

% Initialize the result
dybardth = zeros(ny,nth,N);
dxbardthk = zeros(nx,nth); 
dxhatdthk = zeros(nx,nth);
u1 = u(1,:).';
ytilde1 = ytilde(1,:).';
xbar1 = xbar(1,:).';
for pp = 1:nth
    dybardth(:,pp,1) = (C*dxbardthk(:,pp) + dCdth(:,:,pp)*xbar1 ...
            + dDdth(:,:,pp)*u1 + dbydth(:,pp));
    dxhatdthk(:,pp) = dxbardthk(:,pp) - K*dybardth(:,pp,1) + dKdth(:,:,pp)*ytilde1;
end

% Loop through each sample and parameter to compute sensitivities
for k = 2:N
    for pp = 1:nth
        uavgk = (u(k-1,:) + u(k,:)).'/2;
        xavgk = (xhat(k-1,:) + xbar(k,:)).'/2;
        dxbardthk(:,pp) = Phi*dxhatdthk(:,pp) + Psi*dAdth(:,:,pp)*xavgk ...
            + Psi*dBdth(:,:,pp)*uavgk + Psi*dbxdth(:,pp);
        uk = u(k,:).';
        xbark = xbar(k,:).';
        dybardth(:,pp,k) = (C*dxbardthk(:,pp) + dCdth(:,:,pp)*xbark ...
            + dDdth(:,:,pp)*uk + dbydth(:,pp));
        ytildek = ytilde(k,:).';
        dxhatdthk(:,pp) = dxbardthk(:,pp) - K*dybardth(:,pp,k) + dKdth(:,:,pp)*ytildek;
    end
    
    if k == 3
        % dxbardthk
        % dybardth(:,:,k)
        % dxhatdthk
    end

end

% % Loop through each parameter and sample to compute sensitivities
% dxbardthk = zeros(nx,nth); % initial condition
% dxhatdthk = zeros(nx,nth);
% for k = 1:N
%     for pp = 1:nth
%         uk = u(k,:).';
%         ytildek = ytilde(k,:).';
%         xbark = xbar(k,:).';
%         dybardth(:,pp,k) = (C*dxbardthk(:,pp) + dCdth(:,:,pp)*xbark ...
%             + dDdth(:,:,pp)*uk + dbydth(:,pp));
%         if k == N % no need to compute the following after the Nth sample
%             continue
%         end
%         if k == 2 && pp == 1
%             dybardth(:,pp,k)
%             dxhatdthk(:,pp)
%             test = 0;
%         end
%         dxhatdthk(:,pp) = dxbardthk(:,pp) - K*dybardth(:,pp,k) + dKdth(:,:,pp)*ytildek;
%         uavgk = (uk + u(k+1,:).')/2;
%         xavgk = (xbark + xhat(k+1,:).')/2;
%         dxbardthk(:,pp) = Phi*dxhatdthkpp + Psi*dAdth(:,:,pp)*xavgk ...
%             + Psi*dBdth(:,:,pp)*uavgk + Psi*dbxdth(:,pp); % dxbardth(k+1)
%     end
% end

end

