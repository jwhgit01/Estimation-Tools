close all
clear
clc
addpath(genpath('../'))
load ContinuousTimeNonlinearSystem.mat t x z Sigma R params tSim xSim

% Process noise PSD
Q = Sigma*Sigma';

% UKF tuning parameters
alpha = 0.1;
beta = 2;
kappa = 0;

% Create the UKF
UKFCD = unscentedKalmanFilterCD(@driftModel,@diffusionModel,@measurementModel,Q,R,alpha,beta,kappa,3);

% Initial Condition
xhat0 = [0;0;0];
P0 = 10*eye(3);

% Run the filter on the data
[xhat,P,nu,epsnu] = UKFCD.simulate(t,z,[],xhat0,P0,params);

% Plot the state estimates
figure
hold on
plot(tSim,xSim)
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