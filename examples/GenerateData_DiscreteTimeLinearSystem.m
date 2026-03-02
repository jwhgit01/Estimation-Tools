%% GenerateData_DiscreteTimeLinearSystem
%
% Copyright (c) 2026 Jeremy W. Hopwood. All rights reserved.
%
% This script generates data to perform Kalman filering on the discrete
% time linear system
%
%                   x(k+1) = F*x(k) + Gamma*w(k)
%                     z(k) = C*x(k) + v(k)
%
close all
clear
clc

% Repeatable results
rng('default')

% Dimensions
nx = 4;
nw = 2;
nz = 2;

% System matrices
F = [1, 0.1, 0,   0;
     0,   1, 0,   0;
     0,   0, 1, 0.1;
     0,   0, 0,   1];
Gamma = [0, 0;
         1, 0;
         0, 0;
         0, 1];
C = [1, 0, 0, 0;
     0, 0, 1, 0];

% Process noise covariance
Q = diag([0.1 0.1]);

% Measurement noise covariance
R = diag([0.6 1.2]);


%% Discrete-Time Simulation

% Number of samples (including initial condition)
N = 101;

% Initial condition
x0 = [10; 0; -10; 0];

% Initialize result
x = zeros(N,nx);
x(1,:) = x0.';
z = nan(N,nz);

% Loop through each sample, propogate state, and compute measured output
sqrtQ = chol(Q);
sqrtR = chol(R);
for k = 1:N-1
    x(k+1,:) = (F*x(k,:).' + Gamma*sqrtQ*randn(nw,1)).';
    z(k+1,:) = (C*x(k+1,:).' + sqrtR*randn(nz,1)).';
end


%% Save and plot results

% Save data
vars = {'N','nx','nw','nz','x','z','F','Gamma','C','Q','R','x0'};
save('./data/ExampleData_DiscreteTimeLinearSystem.mat',vars{:})

% Plot states
figure
plot((0:N-1).',x)

% Plot measurements
figure
plot((0:N-1).',z)
