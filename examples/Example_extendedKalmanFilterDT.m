close all
clear
clc
addpath(genpath('../'))

% User options
ContinuousTimeSimulation = true;

if ContinuousTimeSimulation

    % Load data
    load ContinuousTimeNonlinearSystem.mat t x z Sigma R params tSim xSim

    % Equivalent discrete-time process noise covariance
    dt = median(diff(t));
    Q = Sigma*Sigma'/dt;

    % Create the EKF
    EKF = extendedKalmanFilterDT(@driftModel,@diffusionModel,@measurementModel,Q,R);

else

    % Load data
    load DiscreteTimeNonlinearSystem.mat t x z w Q R params

    % Create the EKF
    EKF = extendedKalmanFilterDT(@differenceEquation,[],@measurementModel,Q,R);

end

% Initial Condition
xhat0 = [0;0;0];
P0 = 10*eye(3);

% Run the filter on the data
[xhat,P,nu,epsnu,sigdig] = EKF.simulate(t,z,[],xhat0,P0,params);

% Plot the state estimates
figure
hold on
plot(t,x)
plot(t,xhat,'--')
hold off
grid on
legend('$x_1$','$x_2$','$x_3$','$\hat{x}_1$','$\hat{x}_2$','$\hat{x}_3$','Interpreter','LaTeX')

% Plot the innovation statistic
figure
hold on
plot(t,epsnu)
hold off
grid on