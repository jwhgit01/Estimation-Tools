classdef stochasticDifferentialEquation
%stochasticDifferentialEquation
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This class defines implements the stochastic differential equation (SDE)
%
%               dx = f(t,x,u)*dt + D(t,x,u)*Sigma*dW                    (1)
%
% where x is the nx x 1 state vector, u is the nu x 1 input vector, f is
% the drift vector field, D is the diffusion matrix field, Sigma*Sigma' is
% the infinitesimal covariance of the nw x 1 Wiener process Sigma*W. This
% SDE is also often written as
%
%                 dx/dt = f(t,x,u) + D(t,x,u)*wtil                      (2)
%
% where wtil is zero-mean, continuous-time, Gaussian, white noise with
% power spectral density Sigma*Sigma'. It is related to the Wiener process,
% W, by the diffusive scaling constant sqrt(dt) such that
%
%    var(Delta W) = dt ==> Delta W = sqrt(dt) ==> wtil = Sigma*dW/dt    (3)
%
% Properties
%   Drift
%   Diffusion
%   Sigma
%
% Methods
%   stochasticDifferentialEquation
%   nonlinearDynamics
%   simulate
%

properties
    % The function handle that defines the drift vector field of (1). It
    % must be in the form [f,A] = drift(t,x,u,params), where f is the
    % value of the drift fector field at (t,x,u), A is the Jacobian of f at
    % (t,x,u), t is the current time, x is the state vector, u is the input
    % vector, and params is a struct of parameters.
    Drift

    % The function handle that defines the diffusion matrix field of (1). 
    % It must be in the form D = diffusion(t,x,u,params), where D is the
    % value of the diffusion matrix field at (t,x,u), t is the current
    % time, x is the state vector, u is the input vector, and params is a
    % struct of parameters.
    Diffusion

    % The nw x nw matrix such that Sigma*Sigma' is the infinitesimal
    % covariance of the Wiener process Sigma*W, which is also the power
    % spectral density of the continuous-time white noise wtil.
    Sigma
end

methods
    function obj = stochasticDifferentialEquation(Drift,Diffusion,Sigma)
        %stochasticDifferentialEquation Construct an instance of this class
        obj.Drift = Drift;
        obj.Diffusion = Diffusion;
        obj.Sigma = Sigma;
    end

    function [dxdt,A,D] = nonlinearDynamics(obj,t,x,u,wtil,dervflag,params)
        %odefun Defines a function that can be readily simulated using
        % Matlab's ODE solvers a well as used with legacy versions of this
        % toolbox. CAUTION: Simulating this function using ode45 is not
        % mathematically accurate, but can be useful nonetheless.
        if dervflag && isempty(wtil)
            [f,A] = obj.Drift(t,x,u,params);
            D = obj.Diffusion(t,x,u,params);
            dxdt = f;
        elseif dervflag && ~isempty(wtil)
            [f,A] = obj.Drift(t,x,u,params);
            D = obj.Diffusion(t,x,u,params);
            dxdt = f + D*wtil;
        elseif ~dervflag && isempty(wtil)
            f = obj.Drift(t,x,u,params);
            A = [];
            D = [];
            dxdt = f;
        else
            f = obj.Drift(t,x,u,params);
            A = [];
            D = [];
            dxdt = f + obj.Diffusion(t,x,u,params)*wtil;
        end
    end

    function [x,tEM,xEM,wtil] = simulate(obj,t,ufun,x0,dt,params,W)
        %simulate Simulate the SDE (1) using the Euler-Maruyama scheme.
        %
        % Inputs:
        %   t       The N x 1 array of sample times
        %   ufun    The handle for the function u = ufun(t,x)
        %   x0      The nx x 1 initial condition
        %   dt      The integration time step (may be an empty array, [])
        %   params  The struct of parameters passed to the system model
        %   W       (Optional) A pre-generated Wiener process
        %
        % Outputs:
        %   x       The N x nx array of state values at times in t
        %   tEM     The M x 1 array of Euler-Maruyama integration times
        %   xEM     The M x nx array of state values at times in tEM
        %   wtil    The M x nw array of equivalent white noise values
        %
        
        % Dimensions
        nx = size(x0,1);
        nw = size(obj.Sigma,2);
        
        % Start and end times
        t0 = t(1);
        T = t(end);

        % Euler-Maruyama time steps
        if isempty(dt)
            dt = median(diff(t))/10;
        end
        tEM = (t0:dt:T).';
        N = length(tEM);
        sqrtdt = sqrt(dt);
        
        % Initialize result
        xEM = zeros(N,nx);
        xEM(1,:) = x0.';
        wtil = zeros(N,nw);
        
        % Euler-Maruyama scheme
        for k = 2:N

            % State, time, and input at previous time step
            xkm1 = xEM(k-1,:).';
            tkm1 = tEM(k-1);
            if isempty(ufun)
                ukm1 = [];
            else
                ukm1 = ufun(tkm1,xkm1);
            end

            % Drift vector and diffusion matrix
            f = obj.Drift(tkm1,xkm1,ukm1,params);
            D = obj.Diffusion(tkm1,xkm1,ukm1,params);

            % Increment of the Wiener process W
            if nargin < 7 || isempty(W)
                dW = sqrtdt*randn(nw,1);
            else
                dW = (W(k,:)-W(k-1,:)).';
            end

            % Equivalent white noise vector
            wtil(k-1,:) = (obj.Sigma*dW/dt).';

            % Euler-Maruyama integration step
            xEM(k,:) = (xkm1 + f*dt + D*obj.Sigma*dW).';

        end
        
        % Re-sample xEM to times in t
        x = interp1(tEM,xEM,t,"previous");
    end
end

end