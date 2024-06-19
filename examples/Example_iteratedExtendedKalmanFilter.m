close all
clear
clc
addpath(genpath('../'))
load DiscreteTimeNonlinearSystem.mat t x z w Q R params

% Create the EKF
IEKF = iteratedExtendedKalmanFilter(@differenceEquation,[],@measurementModel,Q,R);

% Initial Condition
xhat0 = [11;11;-11];
P0 = 10*eye(3);

% Run the filter on the data
[xhat,P,what] = IEKF.simulate(t,z,[],xhat0,P0,params);

% Plot the state estimates
figure
hold on
plot(t,x)
plot(t,xhat,'--')
hold off
grid on
legend('$x_1$','$x_2$','$x_3$','$\hat{x}_1$','$\hat{x}_2$','$\hat{x}_3$','Interpreter','LaTeX')

% Plot the noise
figure
hold on
plot(t,w)
plot(t,what,'--')
hold off
grid on