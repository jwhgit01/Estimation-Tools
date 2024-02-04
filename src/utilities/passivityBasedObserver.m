classdef (Abstract) passivityBasedObserver < handle
%passivityBasedObserver

properties

    % The positive scalar that lower bounds the convergence rate of the
    % measurable state estimate error dynamics.
    epsilon(1,1) double {mustBePositive} = 1

    % The invertible injection gain matrix for the measurable states.
    L1 double

    % Whether or not the gain functions for the observer are gain-scheduled
    % by the current time, unmeasurable state estimate, and inputs.
    GainSchedule logical

    % The indices of the measureable states.
    yIdx(1,:) {mustBeInteger,mustBePositive}

    % The char array, string, or function handle that specifies the
    % function that computes the system dynamics. Note the Jacobians of the
    % dynamics are not required. See ~/src/templates/nonlindyn_temp.m.
    nonlindyn

    % The parameters array or struct that gets passed to the dynamics.
    params

end % Properties

properties (Access=protected)
    nx(1,1) {mustBeInteger,mustBePositive} % # of states
    ny(1,1) {mustBeInteger,mustBePositive} % # of measurable states
    nx2(1,1) {mustBeInteger,mustBePositive} % # of unmeasurable states
    x2Idx = setdiff(1:obj.nx,obj.yIdx) % unmeasurable states indices
end % Protected properties

methods (Abstract)

    % The injection gain matrix for the unmeasured states. If L2 is
    % not gain-scheduled, y must be the only argument.
    L2t = L2(obj,y,t,x2hat,u)

    % The matrix- or scalar-valued nonlinear growth bounding function for
    % the measurable state error. If Psi is not gain-scheduled, x1tilde, y,
    % and eta2 must be the only arguments.
    Psit = Psi(obj,x1tilde,y,eta2,t,x2hat,u)

    % The matrix- or scalar-valued nonlinear growth bounding function for
    % the unmeasurable state error. If Lambda is not gain-scheduled,
    % x1tilde, y, and eta2 must be the only arguments.
    Lambdat = Lambda(obj,x1tilde,y,eta2,t,x2hat,u)

end % Abstract methods

methods

    function dxhatdt = observerDynamics(obj,t,xhat,y,u)
        
        % Parse the state vector into its measured and un-measured parts
        x1hat = xhat(obj.yIdx,1);
        x2hat = xhat(obj.x2Idx,1);
        
        % Evaluate the gain function L2 at the current measurment
        if obj.GainSchedule
            L2t = obj.L2(y,t,x2hat,u);
        else
            L2t = obj.L2(y);
        end
        
        % Evaluate the system dynamics at the state current estimate
        fhat = feval(obj.nonlindyn,t,xhat,u,[],false,obj.params);
        
        % Evaluate the Matrix-valued injection gain function, K.
        x1tilde = x1hat - y;
        eta2 = x2hat - (L2t/obj.L1)*x1tilde;
        Psit = obj.Psi(x1tilde,eta2,y,u,x2hat);
        Lambdat = obj.Lambda(x1tilde,eta2,y,u,x2hat);
        K = obj.epsilon*eye(obj.ny) + Psit + Lambdat'*Lambdat;
        
        % Compute the observer dynamics, dxhat/dt.
        L = [obj.L1;L2t];
        yd = x1tilde;
        vd = zeros(obj.ny,1);
        v = -K*yd + vd;
        dxhatdt = fhat + L*v;

    end % observerDynamics

end % Concrete methods

end % classdef