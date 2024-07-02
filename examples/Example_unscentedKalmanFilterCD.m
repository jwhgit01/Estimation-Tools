close all
clear
clc
addpath(genpath('../'))

% System on which unscented Kalman filtering is to be performed
system = "Lorenz"; % "RigidBody" or "Lorenz"

if strcmp(system,"RigidBody")
    load ExampleData_RigidBody.mat t x z Sigma R params tSim xSim
    f = @driftModel_RigidBody;
    D = @diffusionModel_RigidBody;
    h = @measurementModel_RigidBody;
    nx = 6;
    nw = 3;
    nz = 6;
elseif strcmp(system,"Lorenz")
    load ExampleData_Lorenz.mat t x z Sigma R params tSim xSim
    f = @driftModel_Lorenz;
    D = @diffusionModel_Lorenz;
    h = @measurementModel_Lorenz;
    nx = 3;
    nw = 1;
    nz = 2;
else
    error('System not recognized')
end

% Process noise PSD
Q = Sigma*Sigma';

% UKF tuning parameters
alpha = 0.1;
beta = 2;
kappa = 3-nx;

% Create the UKF
UKFCD = unscentedKalmanFilterCD(f,D,h,Q,R,alpha,beta,kappa,nx);

% Initial Condition
xhat0 = zeros(nx,1);
P0 = eye(nx);

% Run the filter on the data
[xhat,P,nu,epsnu] = UKFCD.simulate(t,z,[],xhat0,P0,params);

% Plot the state estimates
figure
hold on
plot(tSim,xSim)
plot(t,xhat,'--')
hold off
grid on

% Plot the state estimate error statistic
figure
hold on
plot(t,epsnu)
hold off
grid on

% Plot the state estimate error statistic
epsx = zeros(size(xhat,1),1);
for k = 1:size(xhat,1)
    exk = xhat(k,:) - x(k,:);
    epsx(k) = (exk/P(:,:,k))*exk';
end
figure
hold on
plot(t,epsx)
hold off
grid on