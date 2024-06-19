classdef extendedKalmanFilterCD < stateEstimatorCD
%extendedKalmanFilterCD
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This class defines the continuous-discrete (hybrid) extended Kalman
% filter for the continuous-time nonlinear system
%
%             dx/dt = f(t,x(t),u(t)) + D(t,x(t),u(t))*wtil(t)           (1)
%
% where x is the nx x 1 state vector, u is the nu x 1 input vector, f is
% the drift vector field, D is the diffusion matrix field, and wtil is
% zero-mean, continuous-time, Gaussian, white noise with power spectral
% density Q. It is assumed there are discrete measurements satisfying
%
%                    z(k) = h(t(k),x(t(k))) + v(k)                      (2)
%
% where each v(k) is independently sampled from a Gaussian distribution
% with zero mean and covariance R(k).
%
% Properties:
%   Drift
%   Diffusion
%   MeasurementModel
%   ProcessNoisePSD
%   MeasurementNoiseCovariance
%   nRK
%
% Methods
%   extendedKalmanFilterCD
%   simulate
%   predict
%   correct
%   Q
%   R
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

    % The function handle that defines the measurment model (2). It must be
    % in the form [y,H] = meas(k,x,u,params), where y is the modeleed
    % output of the system H is its Jacobian. Here, k is the sample number,
    % x is the state vector, u is the input vector, and params is a struct
    % of parameters.
    MeasurementModel
end

methods

    function obj = extendedKalmanFilterCD(f,D,h,Q,R)
        %extendedKalmanFilterDT Construct an instance of this class

        % Store properties
        obj.Drift = f;
        obj.Diffusion = D;
        obj.MeasurementModel = h;
        obj.ProcessNoisePSD = Q;
        obj.MeasurementNoiseCovariance = R;

    end % extendedKalmanFilterDT

    function [xhat,P,nu,epsnu,sigdig] = simulate(obj,t,z,u,xhat0,P0,params)
        %simulate This method performs continuous-discrete extended Kalman
        % filtering for a given time history of measurments.
        %
        % Inputs:
        %
        %   t       The N x 1 sample time vector. The first element of t
        %           corresponds to the time of the initial condition.
        %
        %   z       The N x nz time history of measurements. The first
        %           element of z corresponds to sample k=0, which occurs at
        %           time t=0, and thus is not used.
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

            % Display the time periodically
            obj.dispIter(t(k+2));

            % Recall, arrays are 1-indexed, but the initial condition
            % occurs at k=0.
            ik = k + 1;
        
            % Perform the dynamic propagation of the state estimate.
            tk = t(ik);
            tkp1 = t(ik+1);
            xhatk = xhat(ik,:).';
            uk = u(ik,:).';
            Pk = P(:,:,ik);
            [xbarkp1,Pbarkp1] = obj.predict(tk,tkp1,xhatk,uk,Pk,params);
        
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

    function [xbarkp1,Pbarkp1] = predict(obj,tk,tkp1,xhatk,uk,Pk,params)
        %predict State propogation step of the EKF
        
        % Prepare for the Runge-Kutta numerical integration by setting up 
        % the initial conditions and the time step.
        x = xhatk;
        P = Pk;
        t = tk;
        delt = (tkp1-tk)/obj.nRK;
        
        % This loop does one 4th-order Runge-Kutta numerical integration
        % step per iteration.  Integrate the state.  If partial derivatives
        % are to be calculated, then the partial derivative matrices
        % simultaneously with the state.
        for jj = 1:obj.nRK
        
            % Step a
            [f,A] = obj.Drift(t,x,uk,params);
            D = obj.Diffusion(t,x,uk,params);
            Pdot = A*P + P*A' + D*obj.Q(t)*D';
            dxa = f*delt;
            dPa = Pdot*delt;
        
            % Step b
            [f,A] = obj.Drift(t+0.5*delt,x+0.5*dxa,uk,params);
            D = obj.Diffusion(t+0.5*delt,x+0.5*dxa,uk,params);
            Pdot = A*(P+0.5*dPa) + (P+0.5*dPa)*A' + D*obj.Q(t+0.5*delt)*D';
            dxb = f*delt;
            dPb = Pdot*delt;
        
            % Step c
            [f,A] = obj.Drift(t+0.5*delt,x+0.5*dxb,uk,params);
            D = obj.Diffusion(t+0.5*delt,x+0.5*dxb,uk,params);
            Pdot = A*(P+0.5*dPb) + (P+0.5*dPb)*A' + D*obj.Q(t+0.5*delt)*D';
            dxc = f*delt;
            dPc = Pdot*delt;
        
            % Step d
            [f,A] = obj.Drift(t+delt,x+dxc,uk,params);
            D = obj.Diffusion(t+delt,x+dxc,uk,params);
            Pdot = A*(P+dPc) + (P+dPc)*A' + D*obj.Q(t+delt)*D';
            dxd = f*delt;
            dPd = Pdot*delt;
        
            % 4th order Runge-Kutta integration result
            x = x + (dxa + 2*(dxb + dxc) + dxd)/6;
            P = P + (dPa + 2*(dPb + dPc) + dPd)/6;
            t = t + delt;
        
        end
        
        % Assign the results to the appropriate outputs.
        xbarkp1 = x;
        Pbarkp1 = P;

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

end % classdef