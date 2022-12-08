function [xhat,zhat] = hinftyFilterCT(t,y,u,xhat0,L,K,params,varagin)
%hinftyFilterCT 
%
%  Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
%  This function ...
%
% The orginin of the nonlinear system must be the same as that of the
% linear system. 
%
%  Inputs:
%
%    t
%
%    z
%
%   L
%
%   K
%
%   varagin     Option A (LTI): varagin = {A,B,C}
%               Option B (nonlinear): varagin = {fc,hk}
%   
%  
%  Outputs:
%
%    xhat
%
%    xhat
% 

% If the system is given as an LTI state equation, use lsim.
if length(varagin) > 2

    % Parse varagin
    A = varagin{1};
    B = varagin{2};
    C = varagin{3};
    
    % Put closed-loop equations in form for use with lsim.
    uf = [y,u];
    Af = A - K*C;
    Bf = [K,B];
    Cf = L;
    Df = zeros(size(L,1),size(Bf,2));
    sys = ss(Af,Bf,Cf,Df);
    
    % Use lsim to run the filter on the given data.
    [zhat,~,xhat] = lsim(sys,uf,t,xhat0);

% If the system is given as dynamics and measurement model function
% handles, use ode45.
else

    % Get the problem dimensions and initialize the output arrays.
    N = size(t,1);
    nx = size(xhat0,1);
    nz = size(L,1);
    xhat = zeros(N,nx);
    zhat = zeros(N,nz);
    xhat(1,:) = xhat0.';
    zhat(1,:) = (L*xhat0).';

    % number of Runge-Kutta integration steps
    nRK = 10;

    % Parse varagin
    fc = varagin{1};
    hk = varagin{2};
    
    % Perform Runge-Kutta integration of the nonlinear filter equations.
    for k = 1:N-1

        % Get estimate at sample k
        xk = xhat(k,:).';
        tk = t(k,1);
        uk = u(k,:).';
        yk = y(k,:).';
        tkp1 = t(k+1,1);
        delt = (tkp1-tk)/nRK;
        
        % This loop does one 4th-order Runge-Kutta numerical integration
        % step per iteration.
        for jj = 1:nRK
            %
            fa = feval(fc,tk,xk,uk,[],0,params);
            ha = feval(hk,tk,xk,uk,0,0,params);
            dxa = (fa + K*(yk-ha))*delt;
            %
            fb = feval(fc,tk+0.5*delt,xk+0.5*dxa,uk,[],0,params);
            hb = feval(hk,tk+0.5*delt,xk+0.5*dxa,uk,0,0,params);
            dxb = (fb + K*(yk-hb))*delt;
            %
            fc = feval(fc,tk+0.5*delt,xk+0.5*dxb,uk,[],0,params);
            hc = feval(hk,tk+0.5*delt,xk+0.5*dxb,uk,0,0,params);
            dxc = (fc + K*(yk-hc))*delt;
            %
            fd = feval(fc,tk+delt,xk+dxc,uk,[],0,params);
            hd = feval(hk,tk+delt,xk+dxc,uk,0,0,params);
            dxd = (fd + K*(yk-hd))*delt;
            %
            xk = xk + (dxa + 2*(dxb + dxc) + dxd)/6;
            tk = tk + delt;
        end
        xhat(k+1,:) = xk.';
        zhat(k+1,:) = (L*xk).';
        
    end

end

end