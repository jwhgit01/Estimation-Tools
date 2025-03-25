classdef unscentedKalmanFilterCD < stateEstimatorCD
%unscentedKalmanFilterCD
%
% Copyright (c) 2024 Jeremy W. Hopwood. All rights reserved.
%
% This class defines the continuous-discrete (hybrid) unscented Kalman
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
% with zero mean and covariance R(k). This filter is implemented and
% referenced by equation number using doi.org/10.1109/TAC.2007.904453
%
% Properties:
%   Drift
%   Diffusion
%   MeasurementModel
%   ProcessNoisePSD
%   MeasurementNoiseCovariance
%   nRK
%   alpha
%   beta
%   kappa
%
% Methods
%   unscentedKalmanFilterCD
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
end % public properties

properties (SetAccess=immutable)
    % A scaling parameter determines the spread of the sigma points about
    % xbar. Typically, one chooses 10e-4 <= alpha <= 1.
    alpha(1,1) double {mustBePositive} = 0.1

    % A tuning parameter that incorporates information about the prior
    % distribution of x. The value of beta = 2 is optimal for a Gaussian
    % distribution because it optimizes some type of matching of higher
    % order terms (see Wan and van der Merwe).
    beta(1,1) double {mustBePositive} = 2

    % A secondary scaling parameter. A good value is typically 3-nx.
    kappa(1,1) double = 0
end % visible immutable properties

properties (SetAccess=immutable,Hidden)
    c(1,1) double
    Wm(:,1) double
    Wc(:,1) double
end % hidden immutable properties

properties (Access=private)
    nx(1,1) double % number of states
    ns(1,1) double % number of sigma points
end % private properties

methods

    function obj = unscentedKalmanFilterCD(f,D,h,Q,R,alpha,beta,kappa,nx)
        %unscentedKalmanFilterCD Construct an instance of this class

        % Store properties
        obj.Drift = f;
        obj.Diffusion = D;
        obj.MeasurementModel = h;
        obj.ProcessNoisePSD = Q;
        obj.MeasurementNoiseCovariance = R;
        obj.alpha = alpha;
        obj.beta = beta;
        obj.kappa = kappa;
        obj.nx = nx;
        obj.ns = 2*nx + 1;

        % Compute and store weights
        lambda = obj.alpha^2*(nx+obj.kappa) - nx;
        obj.c = nx + lambda;
        obj.Wm = zeros(2*nx+1,1);
        obj.Wc = zeros(2*nx+1,1);
        obj.Wm(1,1) = lambda/obj.c;
        obj.Wc(1,1) = lambda/obj.c + (1-obj.alpha^2+obj.beta);
        obj.Wm(2:obj.ns,1) = repmat(1/(2*obj.c),2*nx,1);
        obj.Wc(2:obj.ns,1) = repmat(1/(2*obj.c),2*nx,1);

    end % unscentedKalmanFilterCD

    function [xs,Ps,xhat,P,vtil] = smooth(obj,t,z,u,xhat0,P0,params)
        %smooth This method performs continuous-discrete unscented
        % Rauch–Tung–Striebel (RTS) smoothing for a given time history
        % of measurments.
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
        %   xs      The N x nx array that contains the time history of the
        %           smoothed state vector estimates.
        %
        %   Ps      The nx x nx x N array that contains the time history of
        %           the smoothed estimation error covariance.
        %
        %   xhat    The N x nx array that contains the time history of the
        %           state vector estimates.
        %
        %   P       The nx x nx x N array that contains the time history of
        %           the estimation error covariance.
        %

        % Get the problem dimensions and initialize the output arrays.
        N = size(z,1);
        
        % if no inputs, set to a tall empty array
        if isempty(u)
            u = zeros(N,0);
        end
        
        % First, perform unscented Kalman filtering forward in time.
        [xhat,P] = obj.simulate(t,z,u,xhat0,P0,params);
        
        % Initialize the smoothed outputs
        xs = xhat;
        Ps = P;
        vtil = zeros(N,size(obj.Q(0),1));
        
        % Get the covariance and mean at the last sample.
        Psk = P(:,:,N);
        xsk = xhat(N,:).';

        % This loop propagates backwards in time and performs RTS smoothing.
        for k = N-1:-1:1

            % Display the time periodically
            obj.dispIter(t(k+1));
        
            % Recall, arrays are 1-indexed, but the initial condition occurs at k=0
            ik = k+1;
        
            % Smooth the sigma points backwards in time.
            tk = t(ik);
            tkm1 = t(ik-1);
            xhatk = xhat(ik,:).';
            Pk = P(:,:,ik);
            uk = u(ik,:).';
            [xskm1,Pskm1] = obj.rts(tk,tkm1,xsk,uk,Psk,xhatk,Pk,params);

            % Approximate the noise
            dtk = tk - tkm1;
            ukm1 = u(ik-1,:).';
            fkm1 = obj.Drift(tkm1,xskm1,ukm1,params);
            Dkm1 = obj.Diffusion(tkm1,xskm1,ukm1,params);
            vtilkm1 = pinv(Dkm1)*(xsk - fkm1*dtk);
            vtil(ik-1,:) = vtilkm1.';
        
            % Store the mean and covariance
            xs(ik-1,:) = xskm1;
            Ps(:,:,ik-1) = Pskm1;

            % Update xsk and Psk
            xsk = xskm1;
            Psk = Pskm1;
        
        end

    end % smooth

    function [xhat,P,nu,epsnu] = simulate(obj,t,z,u,xhat0,P0,params)
        %simulate This method performs continuous-discrete unscented Kalman
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

        % Get the problem dimensions and initialize the output arrays.
        N = size(z,1);
        nz = size(z,2);
        xhat = zeros(N,obj.nx);
        P = zeros(obj.nx,obj.nx,N);
        nu = zeros(N,nz);
        epsnu = zeros(N,1);
        xhat(1,:) = xhat0.';
        P(:,:,1) = P0;
        
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
            [xhatkp1,Pkp1,nukp1,Skp1] = obj.correct(k+1,zkp1,tkp1,xbarkp1,ukp1,Pbarkp1,params);
            
            % Store the results
            xhat(ik+1,:) = xhatkp1.';
            P(:,:,ik+1) = Pkp1;
            nu(ik+1,:) = nukp1.';
            epsnu(ik+1,:) = nukp1'*(Skp1\nukp1);

        end

    end % simulate

    function [xbarkp1,Pbarkp1] = predict(obj,tk,tkp1,xhatk,uk,Pk,params)
        %predict State propogation step of the UKF
        
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
            [dxbardt,dPdt] = obj.sigmaPointDynamicsCT(t,x,P,uk,params);
            dxbara = dxbardt*delt;
            dPa = dPdt*delt;
        
            % Step b
            [dxbardt,dPdt] = obj.sigmaPointDynamicsCT(t+0.5*delt,x+0.5*dxbara,P+0.5*dPa,uk,params);
            dxbarb = dxbardt*delt;
            dPb = dPdt*delt;
        
            % Step c
            [dxbardt,dPdt] = obj.sigmaPointDynamicsCT(t+0.5*delt,x+0.5*dxbarb,P+0.5*dPb,uk,params);
            dxbarc = dxbardt*delt;
            dPc = dPdt*delt;
        
            % Step d
            [dxbardt,dPdt] = obj.sigmaPointDynamicsCT(t+delt,x+dxbarc,P+dPc,uk,params);
            dxbard = dxbardt*delt;
            dPd = dPdt*delt;
        
            % 4th order Runge-Kutta integration result
            x = x + (dxbara + 2*(dxbarb + dxbarc) + dxbard)/6;
            P = P + (dPa + 2*(dPb + dPc) + dPd)/6;
            t = t + delt;
        
        end
        
        % Assign the results to the appropriate outputs.
        xbarkp1 = x;
        Pbarkp1 = P;

    end % predict

    function [xhatkp1,Pkp1,nukp1,Skp1] = correct(obj,kp1,zkp1,tkp1,xbarkp1,ukp1,Pbarkp1,params)
        %predict Measurement correction step of the EKF

        % Compute lower triangular Cholesky factor of Pbar(k+1) satisfying
        % Eq. (15).
        [Sbarkp1tp,flag] = chol(Pbarkp1);
        Sbarkp1 = Sbarkp1tp';
        if flag > 0
            warning('Singular Sbar(k+1) at sample k=%i',kp1);
            Sbarkp1 = chol(Pbarkp1 + 1e-3*eye(size(Pbarkp1)))';
        end
        
        % Construct matrices of sigma point outputs.
        [Xbarkp1,~,Ybarkp1] = obj.sigmaPointsCT(tkp1,xbarkp1,ukp1,Sbarkp1,params);

        % Mean output of the system
        ybarkp1 = Ybarkp1*obj.Wm;

        % Compute Kalman gain based on sigma point innovation covariances
        nz = size(zkp1,1);
        Skp1 = zeros(nz,nz);
        Ckp1 = zeros(obj.nx,nz);
        for ii = 1:obj.ns
            Skp1 = Skp1 + obj.Wc(ii,1)*(Ybarkp1(:,ii)-ybarkp1)*(Ybarkp1(:,ii)-ybarkp1)';
            Ckp1 = Ckp1 + obj.Wc(ii,1)*(Xbarkp1(:,ii)-xbarkp1)*(Ybarkp1(:,ii)-ybarkp1)';
        end
        Skp1 = Skp1 + obj.R(kp1);
        Kkp1 = Ckp1/Skp1;

        % State estimate
        nukp1 = zkp1 - ybarkp1;
        xhatkp1 = Xbarkp1(:,1) + Kkp1*nukp1;
        Pkp1 = Pbarkp1 - Kkp1*Skp1*Kkp1';

    end % correct

    function [xskm1,Pskm1] = rts(obj,tk,tkm1,xsk,uk,Psk,xhatk,Pk,params)
        %rts RTS smoothing step of the UK smoother
        
        % Prepare for the Runge-Kutta numerical integration by setting up 
        % the initial conditions and the time step.
        xs = xsk;
        Ps = Psk;
        t = tk;
        dt = (tkm1-tk)/obj.nRK;

        % This loop does one 4th-order Runge-Kutta numerical integration
        % step per iteration.  Integrate the state.  If partial derivatives
        % are to be calculated, then the partial derivative matrices
        % simultaneously with the state.
        for jj = 1:obj.nRK
        
            % Step a
            [dxsdt,dPsdt] = obj.sigmaPointSmootherDynamicsCT(t,xs,Ps,xhatk,Pk,uk,params);
            dxsa = dxsdt*dt;
            dPsa = dPsdt*dt;
        
            % Step b
            [dxsdt,dPsdt] = obj.sigmaPointSmootherDynamicsCT(t+0.5*dt,xs+0.5*dxsa,Ps+0.5*dPsa,xhatk,Pk,uk,params);
            dxsb = dxsdt*dt;
            dPsb = dPsdt*dt;
        
            % Step c
            [dxsdt,dPsdt] = obj.sigmaPointSmootherDynamicsCT(t+0.5*dt,xs+0.5*dxsb,Ps+0.5*dPsb,xhatk,Pk,uk,params);
            dxsc = dxsdt*dt;
            dPsc = dPsdt*dt;
        
            % Step d
            [dxsdt,dPsdt] = obj.sigmaPointSmootherDynamicsCT(t+dt,xs+dxsc,Ps+dPsc,xhatk,Pk,uk,params);
            dxsd = dxsdt*dt;
            dPsd = dPsdt*dt;
        
            % 4th order Runge-Kutta integration result
            xs = xs + (dxsa + 2*(dxsb + dxsc) + dxsd)/6;
            Ps = Ps + (dPsa + 2*(dPsb + dPsc) + dPsd)/6;
            t = t + dt;
        
        end
        
        % Assign the results to the appropriate outputs.
        xskm1 = xs;
        Pskm1 = Ps;

    end % rts

end % public methods

methods (Access=private)

    function [dxbardt,dPdt] = sigmaPointDynamicsCT(obj,t,xhat,P,u,params)
        %sigmaPointDynamicsCT This method computes the dynamics of the
        % continuous-time, unscented Kalman filter sigma points.
        
        % Compute the lower triangular Cholesky factor of P(t) satisfying
        % Eq. (15).
        [Sxtp,flag] = chol(P);
        Sx = Sxtp';
        if flag > 0
            warning('Singular S(t) at time t=%0.2f',t);
            Sx = chol(P + 1e-3*eye(size(P)))';
        end
        
        % Construct matrices of sigma points, their dynamics, and outputs.
        [Xk,fX,hX] = obj.sigmaPointsCT(t,xhat,u,Sx,params);
        
        % Get the necessary dimensions.
        nz = size(hX,1);

        % Efficiently compute necessary covariances using Eqs. (36) & (37).
        mx = zeros(obj.nx,1);
        mf = zeros(obj.nx,1);
        mh = zeros(nz,1);
        for ii = 1:obj.ns
            mx = mx + obj.Wm(ii,1)*Xk(:,ii);
            mf = mf + obj.Wm(ii,1)*fX(:,ii);
            mh = mh + obj.Wm(ii,1)*hX(:,ii);
        end
        XWfXtr = zeros(obj.nx,obj.nx);
        XWhXtr = zeros(obj.nx,nz);
        for ii = 1:obj.ns
            XWfXtr = XWfXtr + obj.Wc(ii,1)*(Xk(:,ii)-mx)*(fX(:,ii)-mf)';
            XWhXtr = XWhXtr + obj.Wc(ii,1)*(Xk(:,ii)-mx)*(hX(:,ii)-mh)';
        end
        
        % compute the process noise diffusion matrix
        D = obj.Diffusion(t,xhat,u,params);
        
        % Compute the covariance dynamics using Eq. (34)
        dPdt = XWfXtr + XWfXtr' + D*obj.Q(t)*D';
        
        % Compute the dynamics of the mean using Eq. (34)
        dxbardt = mf;

    end % sigmaPointDynamicsCT

    function [dxsdt,dPsdt] = sigmaPointSmootherDynamicsCT(obj,t,xs,Ps,xhat,P,u,params)
        %sigmaPointSmootherDynamicsCT This method computes the dynamics of the
        % continuous-time, unscented Kalman filter sigma points.
        
        % Compute the lower triangular Cholesky factor of P(t) satisfying
        % Eq. (15).
        [Sxtp,flag] = chol(P);
        Sx = Sxtp';
        if flag > 0
            warning('Singular S(t) at time t=%0.2f',t);
            Sx = chol(P + 1e-3*eye(size(P)))';
        end
        
        % Construct matrices of sigma points, their dynamics, and outputs.
        [Xk,fX] = obj.sigmaPointsCT(t,xhat,u,Sx,params);

        % Efficiently compute necessary covariances.
        mx = zeros(obj.nx,1);
        mf = zeros(obj.nx,1);
        for ii = 1:obj.ns
            mx = mx + obj.Wm(ii,1)*Xk(:,ii);
            mf = mf + obj.Wm(ii,1)*fX(:,ii);
        end
        XWfXtr = zeros(obj.nx,obj.nx);
        for ii = 1:obj.ns
            XWfXtr = XWfXtr + obj.Wc(ii,1)*(Xk(:,ii)-mx)*(fX(:,ii)-mf)';
        end
        
        % Process noise diffusion matrix
        D = obj.Diffusion(t,xhat,u,params);

        % Compute the smoothing gain.
        Qc = obj.Q(t);
        G = (XWfXtr' + D*Qc*D')/P;
        
        % Compute the smoothed estimate dynamics.
        dxsdt = mf + G*(xs-xhat);
        
        % Compute the covariance dynamics.
        dPsdt = G*Ps + Ps*G' - D*Qc*D';

    end % sigmaPointSmootherDynamicsCT

    function [Xk,fX,hX] = sigmaPointsCT(obj,tk,xk,uk,Sx,params)
        %sigmaPointsCT This method computes matrices of sigma points, their
        % dynamics, and outputs given the perturbation sqrt(c)*chol(P)'. 
        
        % Evaluate the dynamics and output at the given estimate, x.
        fX0 = obj.Drift(tk,xk,uk,params);
        hX0 = obj.MeasurementModel(tk,xk,uk,params);
        
        % dimensions
        nz = size(hX0,1);

        % Square root of c
        sqrtc = sqrt(obj.c);
        
        % Construct matrices of sigma points, their dynamics, and outputs.
        Xk = zeros(obj.nx,obj.ns);
        Xk(:,1) = xk;
        fX = zeros(obj.nx,obj.ns);
        fX(:,1) = fX0;
        hX = zeros(nz,obj.ns);
        hX(:,1) = hX0;
        for ii = 2:(obj.nx+1)
            Xk(:,ii) = xk + sqrtc*Sx(:,ii-1);
            fX(:,ii) = obj.Drift(tk,Xk(:,ii),uk,params);
            hX(:,ii) = obj.MeasurementModel(tk,Xk(:,ii),uk,params);
        end
        for ii = (obj.nx+2):obj.ns
            Xk(:,ii) = xk - sqrtc*Sx(:,ii-1-obj.nx);
            fX(:,ii) = obj.Drift(tk,Xk(:,ii),uk,params);
            hX(:,ii) = obj.MeasurementModel(tk,Xk(:,ii),uk,params);
        end
        
    end

end % private methods

end % classdef