%% Generate example data for a nonlinear stochastic differential equation
close all
clear
clc
addpath('../Models')
addpath('../../src/dynamics')

% Sample times
T = 10;
dt = 0.01;
t = (0:dt:T).';
N = length(t);

% Dimensions
nx = 3;
nv = 1;
nz = 2;

% System parameters
params.alpha = 8/3;
params.beta = 10;
params.gamma = 1;

% Square root of process noise PSD
Sigma = 30;

% Initial condition
x0 = [10; 10; -10];

% Create an SDE using the example dift and diffusion models
SDE = stochasticDifferentialEquation(@driftModel,@diffusionModel,Sigma);

% Repeatable results
rng('default')

% Simulate the SDE using the Euler-Maruyama scheme
dtEM = 1e-3;
[x,tSim,xSim,wtil] = SDE.simulate(t,[],x0,dtEM,params);

% Compare with ode45
Nv = length(tSim);
wtilfun = @(t) interp1(tSim,wtil,t,'previous');
odefun = @(t,x) SDE.nonlinearDynamics(t,x,[],wtilfun(t),0,params);
[tode,xode] = ode45(odefun,t,x0);

% Measurment noise
R = diag([0.2,0.1]);
v = (chol(R)*randn(nz,N)).';

% Corrupt measurements with noise
y = zeros(N,nz);
for k = 1:N
    y(k,:) = measurementModel(t(k),x(k,:).',[],params).';
end
z = y + v;

% Save results
vars = {'t','x','y','z','wtil','v','Sigma','R','params','tSim','xSim','SDE'};
save('ContinuousTimeNonlinearSystem.mat',vars{:})

% Plot measurements
figure
hold on
plot(t,y)
plot(t,z)
hold off

% Plot states
figure
hold on
plot(tSim,xSim)
plot(tode,xode,'--')
hold off

