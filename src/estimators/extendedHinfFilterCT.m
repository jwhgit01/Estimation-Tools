classdef extendedHinfFilterCT
%
% Copyright (c) 2025 Jeremy W. Hopwood. All rights reserved.
%
% This class defines the continuous-time extended H_\infty filter for the
% continuous-time nonlinear system
%
%                   dx/dt = f(t,x,u) + B(t,x,u)*w                      (1)
%
% where x is the nx x 1 state vector, u is the nu x 1 input vector, and 
% w \in L_2 is an exogenous disturbance. Assume continuous-time
% measurements
%
%                    y = h(t,x,u) + D(t,x,u)*w                          (2)
%
% are available and that B'*D = 0 --- that is, the process and measurement
% noise are "independent." The performance output is taken to be
%
%                    z = L*(x - xhat)                                   (3)
%
% where xhat is the estimate of x and L is a matrix. The extended H_\infty
% filter aims to approximately solve the following problem:
%
%   Find a filtering gain matrix K(t,xhat,u) for the filter
%
%           dxhat/dt = f(t,xhat,u) + K(t,xhat,u)*(y - h(t,xhat,u))      (4)
%
%   and a positive constant \gamma such that
%
%  \|z\|_{L_2}^2 \leq \gamma^{-2}*( e(0)'*P(0)^{-1}*e(0) + \|w\|_{L_2}^2 ) 
%
%   where e = x - xhat and P(0) is a symmetric positive definite matrix.
% 
%
% Public Properties:
%   Drift
%   ProcessDiffusion
%   MeasurementModel
%   MeasurementDiffusion
%   L
%   gamma
%   DisplayPeriod
%
% Public Methods:
%   extendedHinfFilterCT
%   filter
%

properties
    % The function handle that defines the drift vector field of (1). It
    % must be in the form [f,A] = drift(t,x,u,params), where f is the
    % value of the drift fector field at (t,x,u), A is the Jacobian of f at
    % (t,x,u), t is the current time, x is the state vector, u is the input
    % vector, and params is a struct of parameters.
    Drift

    % The function handle that defines the matrix function B in (1). It
    % must be in the form B = diffusion(t,x,u,params).
    Diffusion

    % The function handle that defines the measurment model (2). It must be
    % in the form [y,H] = meas(t,x,u,params), where y is the modeled
    % output of the system H is its Jacobian.
    MeasurementModel

    % The function handle that defines the matrix function D in (2). It
    % must be in the form D = noise(t,x,u,params).
    MeasurementDiffusion

    % The matrix L that defined the performance output z = L*(x - xhat). It
    % may be also given as a scalar value.
    L {mustBeNumeric} = 1

    % The performance measure, gamma. 
    gamma {mustBePositive} = 1

    % A positive number indicating how frequent the sample number should
    % be displayed. If zero, sample times are not shown.
    DisplayPeriod(1,1) {mustBeNonnegative} = 0

    % Interpolation method for approximating continuous-time measurements
    % and inputs.
    InterpMethod {mustBeMember(InterpMethod,{'linear','previous','pchip'})} = 'linear'

end % public properties

properties(Access=private)
    A
    B
    C
    D
    t
    u
    y
    params
    n
end

methods

    function obj = extendedHinfFilterCT(f,B,h,D)
        %extendedHinfFilterCT Construct an instance of this class

        % Store properties
        obj.Drift = f;
        obj.Diffusion = B;
        obj.MeasurementModel = h;
        obj.MeasurementDiffusion = D;

    end % extendedHinfFilterCT

    function [xhat,P] = filter(obj,t,y,u,xhat0,P0,params)
        %simulate Perform continuous-time extended H_infinity filtering on
        % a given time history of measurements and inputs.
        %
        % Inputs:
        %
        %   t       The Nx1 sample time vector. The first element of t
        %           corresponds to the time of the initial condition.
        %
        %   y       The N x ny time history of measurements. These
        %           measurements will be interpolated according to the
        %           InterpMethod property.
        %
        %   u       The N x nu time history of system inputs (optional). If
        %           not applicable set to an empty array, []. These
        %           measurements will be interpolated according to the
        %           InterpMethod property.
        %
        %   xhat0   The nx x 1 initial state estimate.
        %
        %   P0      The nx x nx symmetric positive definite matrix that
        %           weights the initial estimation error on the performance
        %           measure. It functions like the initial state estimation
        %           error covariance matrix of a Kalman filter.
        %
        %   params  An array or struct of constants that get passed to the
        %           model functions.
        %  
        % Outputs:
        %
        %   xhat    The N x nx array that contains the time history of the
        %           state vector estimates.
        %
        %   P       The nx x nx x N array that contains the time history of
        %           the matrix P.
        %

        % Get the problem dimensions and initialize the output arrays.
        N = size(y,1);
        nx = size(xhat0,1);
        obj.n = nx;
        xhat = zeros(N,nx);
        P = zeros(nx,nx,N);
        xhat(1,:) = xhat0.';
        P(:,:,1) = P0;
        
        % if no inputs, set to a tall empty array
        if isempty(u)
            u = zeros(N,0);
        end

        % Privately store time history of inputs and outputs and parameters
        obj.t = t;
        obj.u = u;
        obj.y = y;
        obj.params = params;

        % Linearize about the initial estimate to set the private 
        % properties A, B, C, and D.
        [~,obj.A] = obj.Drift(t(1),xhat0,u(1,:).',params);
        obj.B = obj.Diffusion(t(1),xhat0,u(1,:).',params);
        [~,obj.C] = obj.MeasurementModel(t(1),xhat0,u(1,:).',params);
        obj.D = obj.MeasurementDiffusion(t(1),xhat0,u(1,:).',params);
        
        % The continuous-time integration of the filter is broken up into
        % N-1 steps determined by the sample times t. This is done so that
        % the linearization of the model about the current estimate is not
        % done for intermediate runge-kutta integration steps. The private
        % properties A, B, C, and D store this linearization, which is
        % updated at the end of each iteration of the following loop.
        for k = 1:N-1

            % Display the time periodically if desired
            obj.dispIter(t(k));

            % Time span to integrate for current step
            tspank = [t(k) t(k+1)];

            % Augmented state at current time step
            xhatk = xhat(k,:).';
            Pk = P(:,:,k);
            xak = [xhatk; Pk(:)];

            % Integrate one time step, keeping the linearization constant
            [tkHist,xakHist] = ode45(@obj.FilterDynamics,tspank,xak);

            % Keep only the final state at t(k+1)
            N = size(tkHist,1);
            xhatkp1 = xakHist(N,1:nx).';
            xhat(k+1,:) = xhatkp1.';
            Pkp1 = reshape(xakHist(N,nx+1:nx^2+nx),[nx nx]);
            P(:,:,k+1) = Pkp1;

            % Re-linearize about new state estimate
            [~,obj.A] = obj.Drift(t(k+1),xhatkp1,u(k+1,:).',params);
            obj.B = obj.Diffusion(t(k+1),xhatkp1,u(k+1,:).',params);
            [~,obj.C] = obj.MeasurementModel(t(k+1),xhatkp1,u(k+1,:).',params);
            obj.D = obj.MeasurementDiffusion(t(k+1),xhatkp1,u(k+1,:).',params);

        end

    end % filter

end % public methods

methods(Access=private)

    function xadot = FilterDynamics(obj,t,xa)
        %FilterDynamics Augmented filter dynamics
        
        % Parse the augmented state vector
        xhat = xa(1:obj.n,1);
        P = reshape(xa(obj.n+1:end,1),[obj.n obj.n]);

        % Interpolate to get current inputs and outputs
        ut = obj.InputFun(t);
        yt = obj.OutputFun(t);

        % Output predication
        yhat = obj.MeasurementModel(t,xhat,ut,obj.params);

        % Filter gain
        R = obj.D*obj.D';
        K = P*obj.C'/R;

        % State estimate dynamics
        xhatdot = obj.Drift(t,xhat,ut,obj.params) + K*(yt - yhat);

        % Riccati equation under the assumption that B'*D = 0
        Pdot = obj.A*P + P*obj.A' ...
            + P*( (1/obj.gamma^2)*(obj.L'*obj.L) - obj.C'*(R\obj.C) )*P ...
            + obj.B*obj.B';

        % Time derivative of augmented filter state vector
        xadot = [xhatdot; Pdot(:)];

    end % FilterDynamics

    function u = InputFun(obj,t)
        %InputFun Interpolate to get the input, u, at time t
        u = interp1(obj.t,obj.u,t,obj.InterpMethod).';
    end % InputFun

    function y = OutputFun(obj,t)
        %OutputFun Interpolate to get the output, y, at time t
        y = interp1(obj.t,obj.y,t,obj.InterpMethod).';
    end % OutputFun

    function dispIter(obj,t)
        %dispIter Periodically display the time current time
        if 0 == mod(t,obj.DisplayPeriod)
            fprintf('t = %0.2f\n',t);
        end
    end % dispIter

end % private methods

end % classdef