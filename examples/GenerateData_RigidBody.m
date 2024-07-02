%% GenerateData_RigidBody
% Simulate a sample path for the randomly forced rigid body described by
% the following Ito SDE:
%
%  d(Theta) = LIB(Theta)*omega*dt                                      (1a)
%  d(omega) = inv(I)*(cross(I*omega,omega))*dt + RIB'(Theta)*Sigma*dW  (1b)
%
close all
clear
clc
addpath models
addpath ../src/dynamics

% Repeatable results
rng('default')

% Dimensions
nx = 6;
nw = 3;
nz = 6;

% System parameters
params.I = [  1,   0, 0.1;
              0, 1.5,   0;
            0.1,   0, 0.5];

%% Continuous-time simulation of SDE

% Simulation and sampling times
T = 10;
dt = 0.01;
t = (0:dt:T).';
N = length(t);
dtEM = 1e-4;
NEM = round(T/dtEM) + 1;

% Pre-generate a standard Wiener process so it can be easily saved
W = zeros(NEM,nw);
for k = 2:NEM
    W(k,:) = W(k-1,:) + sqrt(dtEM)*randn(1,nw);
end

% Square root of process noise PSD
Sigma = diag([1 1 0.5]);

% Create an SDE using the example dift and diffusion models
f = @driftModel_RigidBody;
D = @diffusionModel_RigidBody;
SDE = stochasticDifferentialEquation(f,D,Sigma);

% Initial condition
x0 = [zeros(3,1);randn(3,1)];

% Simulate the SDE using the Euler-Maruyama scheme
[x,tSim,xSim,wtil] = SDE.simulate(t,[],x0,dtEM,params,W);

% Measurment noise
R = 0.01*eye(6);
v = (chol(R)*randn(nz,N)).';

% Corrupt measurements with noise
y = zeros(N,nz);
for k = 1:N
    y(k,:) = measurementModel_RigidBody(k,x(k,:).',[],params).';
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
    yd(k,:) = measurementModel_RigidBody(k,xd(k,:).',[],params).';
end
zd = yd + v;

%% Save and plot results

% Save data
vars = {'t','dt','x','xd','y','z','zd','wtil','W','v','w','Sigma','Q',...
        'R','params','tSim','xSim','SDE'};
save('./data/ExampleData_RigidBody.mat',vars{:})

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
