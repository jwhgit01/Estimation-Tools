% close all
clear
clc
addpath(genpath('../'))

% System on which extended Kalman filtering is to be performed
system = "RigidBody"; % "RigidBody" or "Lorenz"

% Use continuous-time simulation data or true discrete-time system data
DiscreteTimeData = true;

if strcmp(system,"RigidBody")
    load ExampleData_RigidBody.mat t dt x xd z zd Sigma Q R params tSim xSim
    if DiscreteTimeData
        f = @differenceEquation_RigidBody;
        D = [];
    else
        f = @driftModel_RigidBody;
        D = @diffusionModel_RigidBody;
    end
    h = @measurementModel_RigidBody;
    nx = 6;
    nw = 3;
    nz = 6;
elseif strcmp(system,"Lorenz")
    load ExampleData_Lorenz.mat t dt x xd z zd Sigma Q R params tSim xSim
    if DiscreteTimeData
        f = @differenceEquation_Lorenz;
        D = [];
    else
        f = @driftModel_Lorenz;
        D = @diffusionModel_Lorenz;
    end
    h = @measurementModel_Lorenz;
    nx = 3;
    nw = 1;
    nz = 2;
else
    error('System not recognized')
end


% Create the EKF
EKF = extendedKalmanFilterDT(f,D,h,Q,R);

% Initial Condition
xhat0 = zeros(nx,1);
P0 = eye(nx);

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