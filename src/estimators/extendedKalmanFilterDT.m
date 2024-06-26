classdef extendedKalmanFilterDT < stateEstimatorDT
%extendedKalmanFilterDT
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This class defines the discrete-time extended Kalman filter for the
% discrete-time nonlinear system
%
%                   x(k+1) = f(k,x(k),u(k),w(k))                        (1)
%
% where each w(k) is independently samped from a Gaussian distribtion with
% zero mean and covariance Q(k), or the continuous time nonlinear system
%
%             dx/dt = fc(t,x(t),u(t)) + D(t,x(t),u(t))*wtil(t)          (2)
%
% where wtil is continuos-time zero-mean Gaussian white noise with power
% spectral density V. The nonlinear measurement
% model for this system is
%
%                       z(k) = h(k,x(k)) + v(k)                         (4)
%
% where each w(k) is independently samped from a Gaussian distribtion with
% zero mean and covariance R(k).
%
% Properties
%   nonlinearDynamics
%   measurementModel
%   processNoiseCovariance
%   measurementNoiseCovariance
%   nRK
%   differenceEquation
%
% Methods
%   extendedKalmanFilterDT
%   simulate
%   predict
%   correct
%   Q
%   R
%
properties
    % The function handle that defines the difference equation (1). It must
    % be in the form [f,F,G] = system(k,x,u,w,params), where f is the value
    % of fd in (1), F is the Jacobian of fd with respect to x at (k,x,u,0),
    % G is the is the Jacobian of fd with respect to w at (k,x,u,0), k is
    % the current sample number, x is the state vector, u is the input
    % vector, and params is a struct of parameters. This function must
    % return f(k,x(k),u(k),0) when w is given as an empty array, [].
    DifferenceEquation

    % The function handle that defines the drift vector field of (2). It
    % must be in the form [f,A] = drift(t,x,u,params), where f is the
    % value of the drift fector field at (t,x,u), A is the Jacobian of f at
    % (t,x,u), t is the current time, x is the state vector, u is the input
    % vector, and params is a struct of parameters.
    Drift

    % The function handle that defines the diffusion matrix field of (2). 
    % It must be in the form D = diffusion(t,x,u,params), where D is the
    % value of the diffusion matrix field at (t,x,u), t is the current
    % time, x is the state vector, u is the input vector, and params is a
    % struct of parameters.
    Diffusion

    % The function handle that defines the measurment model (4). It must be
    % in the form [y,H] = meas(k,x,u,params), where y is the modeleed
    % output of the system H is its Jacobian. Here, k is the sample number,
    % x is the state vector, u is the input vector, and params is a struct
    % of parameters.
    MeasurementModel

    % The number of intermediate 4th order Runge-Kutta intergation steps
    % between samples. This property is only used if a continuous-time
    % system is given (i.e., DifferenceEquation = []).
    nRK
end

properties (Access=private)
    differenceEquation logical
end

methods

    function obj = extendedKalmanFilterDT(f,D,h,Q,R)
        %extendedKalmanFilterDT Construct an instance of this class.
        %
        % Inputs:
        %   f   The function handle that implements f in (1) or fc in (2).
        %   D   The function handle that implements D in (2) or an empty
        %       array if the system is given as a difference equation (1).
        %   h   The function handle for the measurement model (3).
        %   Q   The covariance of w(k).
        %   R   The covariance of v(k).

        % Determine whether the system is discrete or continuous
        if isempty(D)
            obj.differenceEquation = true;
            obj.DifferenceEquation = f;
        else
            obj.differenceEquation = false;
            obj.Drift = f;
            obj.Diffusion = D;
            obj.nRK = 10;
        end

        % Store remaining properties
        obj.MeasurementModel = h;
        obj.ProcessNoiseCovariance = Q;
        obj.MeasurementNoiseCovariance = R;

    end % extendedKalmanFilterDT

    function [xhat,P,nu,epsnu,sigdig] = simulate(obj,t,z,u,xhat0,P0,params)
        %simulate This method performs discrete-time extended Kalman
        % filtering for a given time history of measurments.
        %
        % Inputs:
        %
        %   t       The N x 1 sample time vector. If the system is given by
        %           a difference equation as specified in the constructor, 
        %           this argument is not used and can be set as an empty
        %           array, []. The first element of t corresponds to the
        %           initial condition at sample k=0.
        %
        %   z       The N x nz time history of measurements. The first
        %           element of z corresponds to sample k=0, and thus is not
        %           used.
        %
        %   u       The N x nu time history of system inputs (optional). If
        %           not applicable set to an empty array, []. The first
        %           element of u corresponds to the input at the initial
        %           condition.
        %
        %   xhat0   The nx x 1 initial state estimate.
        %
        %   P0      The nx x nx symmetric positive definite initial state
        %           estimation error covariance matrix.
        %
        %   params  An array or struct of constants that get passed to the
        %           dynamics model and measurement model functions.
        %  
        % Outputs:
        %
        %   xhat    The N x nx array that contains the time history of the
        %           state vector estimates.
        %
        %   P       The nx x nx x N array that contains the time history of
        %           the estimation error covariance.
        %
        %   nu      The N x nz vector of innovations.
        %
        %   epsnu   The N x 1 vector of the normalized innovation statistic
        %
        %   sigdig  The approximate number of accurate significant decimal
        %           places in the result. This is computed using the
        %           condition number of inovation covariance, S.
        %

        % Get the problem dimensions and initialize the output arrays.
        N = size(z,1);
        nx = size(xhat0,1);
        nz = size(z,2);
        xhat = zeros(N,nx);
        P = zeros(nx,nx,N);
        nu = zeros(N,nz);
        epsnu = zeros(N,1);
        xhat(1,:) = xhat0.';
        P(:,:,1) = P0;
        maxsigdig = -fix(log10(eps));
        sigdig = maxsigdig;
        
        % if no inputs, set to a tall empty array
        if isempty(u)
            u = zeros(N,0);
        end
        
        % This loop performs one model propagation step and one measurement
        % update step per iteration.
        for k = 0:N-2

            % Display the iteration number periodically
            obj.dispIter(k);

            % Recall, arrays are 1-indexed, but the initial condition
            % occurs at k=0.
            ik = k + 1;

            % Get the sample time or number.
            if obj.differenceEquation
                tk = []; % not used
                tkp1 = []; % not used
            else
                tk = t(ik);
                tkp1 = t(ik+1);
            end
        
            % Perform the dynamic propagation of the state estimate.
            xhatk = xhat(ik,:).';
            uk = u(ik,:).';
            Pk = P(:,:,ik);
            [xbarkp1,Pbarkp1] = obj.predict(k,tk,tkp1,xhatk,uk,Pk,params);
        
            % Perform the measurement update of the state estimate.
            ukp1 = u(ik+1,:).';
            zkp1 = z(ik+1,:).';
            [xhatkp1,Pkp1,nukp1,Skp1] = obj.correct(k+1,zkp1,xbarkp1,ukp1,Pbarkp1,params);
            
            % Store the results
            xhat(ik+1,:) = xhatkp1.';
            P(:,:,ik+1) = Pkp1;
            nu(ik+1,:) = nukp1.';
            epsnu(ik+1,:) = nukp1'*(Skp1\nukp1);
        
            % Check the condition number of Skp1 and infer the approximate
            % numerical precision of the resulting estimate.
            sigdigkp1 = maxsigdig - fix(log10(cond(Skp1)));
            if sigdigkp1 < sigdig
                sigdig = sigdigkp1;
            end
        
        end

    end % simulate

    function [xbarkp1,Pbarkp1] = predict(obj,k,tk,tkp1,xhatk,uk,Pk,params)
        %predict State propogation step of the EKF

        % Process noise covariance
        Qk = obj.Q(k);

        % Propogate the system state to the next sample
        if obj.differenceEquation
            [xbarkp1,Fk,Gk] = obj.DifferenceEquation(k,xhatk,uk,[],params);
        else
            [xbarkp1,Fk,Gk] = obj.discretize(tk,tkp1,xhatk,uk,params);
        end

        % Compute the a priori state estimate error covariance
        Pbarkp1 = Fk*Pk*Fk' + Gk*Qk*Gk';

    end % predict

    function [xhatkp1,Pkp1,nukp1,Skp1] = correct(obj,kp1,zkp1,xbarkp1,ukp1,Pbarkp1,params)
        %predict Measurement correction step of the EKF

        % Predicted output of the system
        [zbarkp1,Hkp1] = obj.MeasurementModel(kp1,xbarkp1,ukp1,params);

        % Innovations
        nukp1 = zkp1 - zbarkp1;
        Rkp1 = obj.R(kp1);
        Skp1 = Hkp1*Pbarkp1*Hkp1' + Rkp1;

        % Kalman gain
        Kkp1 = Pbarkp1*Hkp1'/Skp1;

        % State estimate
        nx = size(xbarkp1,1);
        xhatkp1 = xbarkp1 + Kkp1*nukp1;
        Pkp1 = (eye(nx)-Kkp1*Hkp1)*Pbarkp1; % or Pbarkp1 - Kkp1*Skp1*Kkp1'

    end % predict

end % public methods

methods (Access=private)
    function [fk,Fk,Gk] = discretize(obj,tk,tkp1,xk,uk,params)
        
        % Prepare for the Runge-Kutta numerical integration by setting up 
        % the initial conditions and the time step.
        x = xk;
        if nargout > 1
            nx = size(xk,1);
            Dk = obj.Diffusion(tk,xk,uk,params);
            nv = size(Dk,2);
            F = eye(nx);
            G = zeros(nx,nv);
        end
        t = tk;
        dt = (tkp1 - tk)/obj.nRK;

        % This loop does one 4th-order Runge-Kutta numerical integration
        % step per iteration.
        for jj = 1:obj.nRK

            % Step a
            if nargout > 1
                [f,A] = obj.Drift(t,x,uk,params);
                D = obj.Diffusion(t,x,uk,params);
                dFa = (A*F)*dt;
                dGa = (A*G+D)*dt; 
            else
                f = obj.Drift(t,x,uk,params);
            end
            dxa = f*dt;
        
            % Step b
            if nargout > 1
                [f,A] = obj.Drift(t+0.5*dt,x+0.5*dxa,uk,params);
                D = obj.Diffusion(t+0.5*dt,x+0.5*dxa,uk,params);
                dFb = (A*F)*dt;
                dGb = (A*G+D)*dt; 
            else
                f = obj.Drift(t+0.5*dt,x+0.5*dxa,uk,params);
            end
            dxb = f*dt;
        
            % Step c
            if nargout > 1
                [f,A] = obj.Drift(t+0.5*dt,x+0.5*dxb,uk,params);
                D = obj.Diffusion(t+0.5*dt,x+0.5*dxb,uk,params);
                dFc = (A*F)*dt;
                dGc = (A*G+D)*dt; 
            else
                f = obj.Drift(t+0.5*dt,x+0.5*dxb,uk,params);
            end
            dxc = f*dt;
        
            % Step d
            if nargout > 1
                [f,A] = obj.Drift(t+dt,x+dxc,uk,params);
                D = obj.Diffusion(t+dt,x+dxc,uk,params);
                dFd = (A*F)*dt;
                dGd = (A*G+D)*dt;
            else
                f = obj.Drift(t+dt,x+dxc,uk,params);
            end
            dxd = f*dt;
        
            % 4th order Runge-Kutta integration result
            x = x + (dxa + 2*(dxb + dxc) + dxd)/6;
            if nargout > 1
                F = F + (dFa + 2*(dFb + dFc) + dFd)/6;
                G = G + (dGa + 2*(dGb + dGc) + dGd)/6;
            end
            t = t + dt;
        
        end

        % Assign the results to the appropriate outputs.
        fk = x;
        if nargout < 2 
            return
        end
        Fk = F;
        Gk = G;
    end % discretize
end % private methods

end % classdef