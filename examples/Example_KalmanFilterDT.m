%% Example_KalmanFilterDT
%
% Copyright (c) 2026 Jeremy W. Hopwood. All rights reserved.
%
% This script demonstrates how to use the Estimation-Tools repository to
% run the discrete-time Kalman filter on a given time history of
% measurements.
%
close all
clear
clc
addpath(genpath('../'))

% Load data
load ExampleData_DiscreteTimeLinearSystem.mat

% Create the Kalman filter object
KF = kalmanFilterDT(F,[],Gamma,C,[],Q,R);

% Initial Condition
xhat0 = zeros(nx,1);
P0 = eye(nx);

% Run the filter on the data
[xhat,P] = KF.simulate(z,[],xhat0,P0);

% Get the standard deviations of the state estimate components
sigma_xhat = zeros(N,nx);
for k = 1:N
    sigma_xhat(k,:) = diag(P(:,:,k)).';
end

% Plot the state estimates on a single plot
samples = (0:N-1).';
figure
hold on
plot(samples,x)
plot(samples,xhat,'--')
hold off
grid on
xlabel("Sample Number, $k$",Interpreter='LaTeX')
legend('$x_1$','$x_2$','$x_3$','$x_4$','$\hat{x}_1$','$\hat{x}_2$',...
       '$\hat{x}_3$','$\hat{x}_4$',Interpreter='LaTeX')

% Plot the state estimates with 2-sigma error bars
samples2 = [0:N-1, N-1:-1:0].';
upperbound = xhat + 2*sigma_xhat;
lowerbound = xhat - 2*sigma_xhat;
inBetween = [upperbound; flipud(lowerbound)];
figure
tiledlayout(2,2)
for ii = 1:4
    nexttile
    hold on
    plot(samples,x(:,ii),'-k')
    plot(samples,xhat(:,ii),'--r')
    fill(samples2,inBetween(:,ii),"k",FaceAlpha=0.1,EdgeColor='none')
    hold off
    grid on
end
