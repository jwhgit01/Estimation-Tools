%% GenerateData_Lorenz
% Simulate a sample path for the stochastic Lorenz system descrtibed by
% the following Ito SDE:
%
%                   dX = alpha*(Y - X)*dt                              (1a)
%                   dY = (X*(beta - Z) - Y)*dt                         (1b)
%                   dZ = (-gamma*Z + X*Y)*dt + sigma*dW                (1c)
%
close all
clear
clc
addpath models
addpath ../src/dynamics

% Repeatable results
rng('default')

% Dimensions
nx = 3;
nw = 1;
nz = 2;

% System parameters
params.alpha = 8/3;
params.beta = 10;
params.gamma = 1;

%% Continuous-time simulation of SDE

% Simulation and sampling times
T = 10;
dt = 0.01;
t = (0:dt:T).';
N = length(t);
dtEM = 1e-4;
NEM = round(T/dtEM) + 1;

% Square root of process noise PSD
Sigma = 30;

% Initial condition
x0 = [10; 10; -10];

% Create an SDE using the example dift and diffusion models
f = @driftModel_Lorenz;
D = @diffusionModel_Lorenz;
SDE = stochasticDifferentialEquation(f,D,Sigma);

% Pre-generate a standard Wiener process so it can be easily saved
W = zeros(NEM,nw);
for k = 2:NEM
    W(k,:) = W(k-1,:) + sqrt(dtEM)*randn(1,nw);
end

% Simulate the SDE using the Euler-Maruyama scheme
[x,tSim,xSim,wtil] = SDE.simulate(t,[],x0,dtEM,params,W);

% Measurment noise
R = diag([1,0.5]);
v = (chol(R)*randn(nz,N)).';

% Corrupt measurements with noise
y = zeros(N,nz);
for k = 1:N
    y(k,:) = measurementModel_Lorenz(k,x(k,:).',[],params).';
end
z = y + v;

%% Disrete-time data using a large step size

% Save sample time step
params.dt = dt;

% Simulate using the downsampled Wiener process previously generated
w = interp1(tSim,W,t,"previous");
[~,~,xd] = SDE.simulate(t,[],x0,dt,params,w);

% Discrete-time process noise and covariance
Q = Sigma*Sigma'*dt;

% Corrupt measurements with same noise as the continuous-time case
yd = zeros(N,nz);
for k = 1:N
    yd(k,:) = measurementModel_Lorenz(k,xd(k,:).',[],params).';
end
zd = yd + v;

%% Save and plot results

% Save data
vars = {'t','dt','x','xd','y','z','zd','wtil','W','v','w','Sigma','Q',...
        'R','params','tSim','xSim','SDE'};
save('./data/ExampleData_Lorenz.mat',vars{:})

% Plot states
figure
hold on
plot(tSim,xSim)
plot(t,xd,'--')
hold off

% Plot measurements
figure
hold on
plot(t,y)
plot(t,z)
hold off
