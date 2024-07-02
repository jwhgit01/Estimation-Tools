close all
clear
clc
addpath(genpath('../'))

% User options
ContinuousTimeSimulation = true;

if ContinuousTimeSimulation

    % Load data
    % load ContinuousTimeNonlinearSystem.mat t x z W Sigma R params tSim xSim
    load ExampleData_RigidBody.mat t x z W Sigma R params tSim xSim

    % Equivalent discrete-time process noise covariance
    dt = median(diff(t));
    Q = Sigma*Sigma'*dt;
    Wk = interp1(tSim,W,t,"previous");
    w = [zeros(1,3); diff(Wk,1,1)];

    % Create the IEKF
    IEKF = iteratedExtendedKalmanFilter(@driftModel_RigidBody,@diffusionModel_RigidBody,@measurementModel_RigidBody,Q,R);

else

    % Load data
    load DiscreteTimeNonlinearSystem.mat t x z w Q R params

    % Create the EKF
    IEKF = iteratedExtendedKalmanFilter(@differenceEquation,[],@measurementModel,Q/0.01,R);

end

% Initial Condition
xhat0 = zeros(6,1);
P0 = eye(6);

% Run the filter on the data
[xhat,P,what] = IEKF.simulate(t,z,[],xhat0,P0,params);

% Plot the state estimates
figure
hold on
plot(t,x)
plot(t,xhat,'--')
hold off
grid on
%legend('$x_1$','$x_2$','$x_3$','$\hat{x}_1$','$\hat{x}_2$','$\hat{x}_3$','Interpreter','LaTeX')

% Plot the noise
figure
hold on
plot(t,w,'o-')
plot(t,what,'--x')
hold off
grid on
legend('$w_1$','$w_2$','$w_3$','$\hat{w}_1$','$\hat{w}_2$','$\hat{w}_3$','Interpreter','LaTeX')
