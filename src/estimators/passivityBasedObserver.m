classdef (Abstract) passivityBasedObserver
%passivityBasedObserver

properties

    % The invertible injection gain matrix for the measurable part of the
    % state vector.
    L1

    % The positive scalar that lower bounds the convergence rate of the
    % measurable state estimate error dynamics.
    epsilon(1,1) double {mustBePositive} = 1

    % Whether or not the gain functions for the observer are gain-scheduled
    % by the current time, unmeasurable state estimate, and inputs. The
    % methods L2, Psi, and Lambda should make use of this property.
    gainSchedule logical = false

    % The char array, string, or function handle that specifies the
    % function that computes the system dynamics. Note the Jacobians of the
    % dynamics are not required. See templates/nonlinearDynamics_temp.m.
    nonlinearDynamics

    % An array or stuct of constants that are used in nonlinearDynamics as
    % well as L2, Psi, and Lambda.
    params

end % public properties

properties (Abstract,Constant)

    % The indices of the measureable states in nonlinearDynamics
    yIdx

    % The indices of the unmeasureable states in nonlinearDynamics
    x2Idx

end % abstract constant properties

methods (Abstract)

    % The injection gain matrix for the unmeasured states. If L2 is
    % not gain-scheduled, the arguments t, x2hat, & u should not be used.
    L2t = L2(obj,y,t,x2hat,u)

    % The matrix- or scalar-valued nonlinear growth bounding function for
    % the measurable state error. If Psi is not gain-scheduled, the
    % arguments t, x2hat, & u should not be used.
    Psit = Psi(obj,x1tilde,y,eta2,t,x2hat,u)

    % The matrix- or scalar-valued nonlinear growth bounding function for
    % the unmeasurable state error. If Lambda is not gain-scheduled, the
    % arguments t, x2hat, & u should not be used.
    Lambdat = Lambda(obj,x1tilde,y,eta2,t,x2hat,u)

end % abstract methods

methods

    function dxhatdt = observerDynamics(obj,t,xhat,y,u)
        %observerDynamics The dynamics of the passivity-based observer
        
        % Dimensions
        nx = size(xhat,1);
        ny = size(y,1);

        % Parse the state vector into its measured and un-measured parts
        x1hat = xhat(obj.yIdx,1);
        x2hat = xhat(obj.x2Idx,1);
        
        % Evaluate the gain function L2 at the current measurment
        L2t = obj.L2(y,t,x2hat,u);
        
        % Evaluate the system dynamics at the state current estimate
        fhat = feval(obj.nonlinearDynamics,t,xhat,u,[],false,obj.params);
        f1hat = fhat(obj.yIdx,1);
        f2hat = fhat(obj.x2Idx,1);
        
        % Evaluate the injection gain function, K.
        x1tilde = x1hat - y;
        eta2 = x2hat - (L2t/obj.L1)*x1tilde;
        Psit = obj.Psi(x1tilde,eta2,y,t,x2hat,u);
        Lambdat = obj.Lambda(x1tilde,eta2,y,t,x2hat,u);
        K = obj.epsilon*eye(ny) + Psit + Lambdat'*Lambdat;
        
        % Compute the observer dynamics, dxhat/dt.
        v = -K*x1tilde;
        dxhatdt = zeros(nx,1);
        dxhatdt(obj.yIdx,1) = f1hat + obj.L1*v;
        dxhatdt(obj.x2Idx,1) = f2hat + L2t*v;

    end % observerDynamics

end % concrete methods

end % classdef