function [fk,Fk,Gamk] = c2dNonlinear(xk,uk,vk,tk,tkp1,nRK,fc,dervflag,params)
%
%  Copyright (c) 2002 Mark L. Psiaki.    All rights reserved.  
%                2022 Jeremy W. Hopwood. All rights reserved.
%
%
%  This function derives a nonlinear discrete-time dynamics function
%  for use in a nonlinear difference equation via 4th-order 
%  Runge-Kutta numerical integration of a nonlinear differential
%  equation.  If the nonlinear differential equation takes the
%  form:
%
%           xdot = f(t,x,uk,vk)
%
%  and if the initial condition is x(tk) = xk, then the solution
%  gets integrated forward from time tk to time tkp1 using nRK
%  4th-order Runge-Kutta numerical integration steps in order to
%  compute fk(k,xk,uk,vk) = x(tkp1).  This function can
%  be used in a nonlinear dynamics model of the form:
%
%           xkp1 = fk(k,xk,uk,vk)
%
%  which is the form for use in an extended Kalman filter.
%
%  This function also computes the first partial derivative of 
%  fk(k,xk,uk,vk) with respect to xk, Fk = dfk/dxk, and with
%  respect to vk, Gk = dfk/dvk.
%
%
%  Inputs:
%
%    xk             The state vector at time tk, which is the initial 
%                   time of the sample interval.
%
%    uk             The control vector, which is held constant
%                   during the sample interval from time tk to time
%                   tkp1.
%
%    vk             The discrete-time process noise disturbance vector, 
%                   which is held constant during the sample interval 
%                   from time tk to time tkp1.
%
%    tk             The start time of the numerical integration
%                   sample interval.
%
%    tkp1           The end time of the numerical integration
%                   sample interval.
%
%    nRK            The number of Runge-Kutta numerical integration
%                   steps to take during the sample interval.
%
%    fc             The name of the Matlab .m-file that contains the
%                   function which defines f(t,x,uk,vk). This may be a
%                   character string or a function handle.  For example, if
%                   the continuous-time differential equation model is
%                   contained in the file rocketmodel.m with the function
%                   name rocketmodel, then on input to the present function
%                   fc must equal 'rocketmodel' or @rocketmodel, and the
%                   first line of the file rocketmodel.m must be:
%
%                   function [f,A,Gam] = rocketmodel(t,x,u,vtil,dervflag,params)
%
%                   The function must be written so that fscript defines
%                   xdot as a function of t, x, u, and vtil and so that A
%                   and Gam are the matrix partial derivatives of f with
%                   respect to x and vtil if dervflag = 1.  If dervflag =
%                   0, then these outputs must be empty arrays.
%
%    dervflag       A flag that tells whether (dervflag = 1) or not
%                   (dervflag = 0) the partial derivatives df/dxk and
%                   df/dvk must be calculated. If dervflag = 0, then these
%                   outputs will be empty arrays.
%
%   params          Parameters passed to the system dynamic model function.
%  
%  Outputs:
%
%    fk             The discrete-time dynamics vector function evaluated 
%                   at k, xk, uk, and vk.
%
%    Fk             The partial derivative of fprinted with respect to xk.
%                   This is a Jacobian matrix. It is evaluated and output
%                   only if dervflag = 1. Otherwise, an empty array is
%                   output.
%
%    Gk             The partial derivative of fprinted with respect to vk.
%                   This is a Jacobian matrix. It is evaluated and output
%                   only if dervflag = 1. Otherwise, an empty array is
%                   output.
%

% Prepare for the Runge-Kutta numerical integration by setting up 
% the initial conditions and the time step.
x = xk;
if dervflag == 1
    nx = size(xk,1);
    nv = size(vk,1);
    F = eye(nx);
    Gam = zeros(nx,nv);
end
t = tk;
delt = (tkp1 - tk)/nRK;

% This loop does one 4th-order Runge-Kutta numerical integration step
% per iteration.  Integrate the state.  If partial derivatives are
% to be calculated, then the partial derivative matrices simultaneously
% with the state.
for jj = 1:nRK

    % Step a
    if dervflag == 1
        [f,A,D] = feval(fc,t,x,uk,vk,1,params);
        dFa = (A*F)*delt;
        dGama = (A*Gam+D)*delt; 
    else
        f = feval(fc,t,x,uk,vk,0,params);
    end
    dxa = f*delt;

    % Step b
    if dervflag == 1
        [f,A,D] = feval(fc,t+0.5*delt,x+0.5*dxa,uk,vk,1,params);
        dFb = (A*F)*delt;
        dGamb = (A*Gam+D)*delt; 
    else
        f = feval(fc,t+0.5*delt,x+0.5*dxa,uk,vk,0,params);
    end
    dxb = f*delt;

    % Step c
    if dervflag == 1
        [f,A,D] = feval(fc,t+0.5*delt,x+0.5*dxb,uk,vk,1,params);
        dFc = (A*F)*delt;
        dGamc = (A*Gam+D)*delt; 
    else
        f = feval(fc,t+0.5*delt,x+0.5*dxb,uk,vk,0,params);
    end
    dxc = f*delt;

    % Step d
    if dervflag == 1
        [f,A,D] = feval(fc,t+delt,x+dxc,uk,vk,1,params);
        dFd = (A*F)*delt;
        dGamd = (A*Gam+D)*delt;
    else
        f = feval(fc,t+delt,x+dxc,uk,vk,0,params);
    end
    dxd = f*delt;

    % 4th order Runge-Kutta integration result
    x = x + (dxa + 2*(dxb + dxc) + dxd)/6;
    if dervflag == 1
        F = F + (dFa + 2*(dFb + dFc) + dFd)*(1/6);
        Gam = Gam + (dGama + 2*(dGamb + dGamc) + dGamd)/6;
    end
    t = t + delt;

end

% Assign the results to the appropriate outputs.
fk = x;
if dervflag == 1
    Fk = F;
    Gamk = Gam;
else
    Fk = [];
    Gamk = [];
end

end