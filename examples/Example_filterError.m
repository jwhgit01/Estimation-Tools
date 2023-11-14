% Linear Filter Error Maximum Likelihood Parameter Estimation
%
% Based on Chapter 5: Filter Error Method, "Flight Vehicle System
% Identification - A Time Domain Methodology" by Ravindra V. Jategaonkar.
close all
clear
clc

% Add to path
addpath(genpath('..\src'))
addpath('models')

% Load data
load -ascii ./data/y13aus_da1.asc
data = y13aus_da1;     

% Sampling time and sample times
N = size(data,1); 
dt = 0.05;   
t = (0:dt:N*dt-dt)';

% Measurments: pdot, rdot, ay, p, r
z = [data(:,12) data(:,13) data(:,14) data(:,15) data(:,16)];

% Inputs: aileron, rudder, vk
u = [data(:,17) data(:,18) data(:,19)];

% Initial condition p0, r0
xhat0 = [0; 0];

% Initial output residual covariance
% S0 = diag([0.164516  0.1487645  0.2213919  0.0824749  0.4420580]);
S0 = [];

% Initial parameter estimates
theta0 = [-6.7; 1.83; -0.906; -0.665; -18.3; 0.43; -0.114; -0.66; -2.82;...
          0.0069; -0.64;  1.30; -1.40;   2.79; -0.193; 0.02;  0.02; 0.08;....
          -0.089; 0.08; -0.089; 0.044; -0.0034; -0.0015];
Ptheta0 = [];

% System constants
constants = [];

% Contrained update method
guard = 'Constrained';

% Parameter indices
thetaIdx.A = [1 2; 3 4];
thetaIdx.B = [5 6 7; 8 9 10];
thetaIdx.C = [1 2; 3 4; 11 12; zeros(2,2)];
thetaIdx.D = [5 6 7; 8 9 10; 13 14 15; zeros(2,3)];
thetaIdx.F = [16 0; 0 17];
thetaIdx.G = zeros(5,5);
thetaIdx.bx = [18; 19];
thetaIdx.by = (20:24).';

% Algorithm parameters
feParams.RelativeTolerance = 1e-3;
feParams.MaxIterations = 15;
feParams.FCorrectionIteration = 3;

% Display
dispflag = true;

% Run the filer error algorithm
out = filterError(dt,z,u,xhat0,theta0,S0,Ptheta0,thetaIdx,@ssfun,constants,guard,feParams,dispflag);

% Plot the results
ybar = out.OutputPropogation;
thetaHist = out.ParameterEstimateHistory;
thetaStd = zeros(size(thetaHist));
numIters = out.NumberOfIterations;
for ii = 1:numIters+1
    thetaStd(:,ii) = sqrt(diag(inv(out.FisherInformationMatrixHistory(:,:,ii))));
end

% Conversion factor radians to degrees
r2d = 180/pi;

% Plot time histories of measured and estimated observation variables
figure
subplot(811),plot(t,z(:,1)*r2d, t,ybar(:,1)*r2d,'r--'); grid; ylabel('pdot (°/s2)');
subplot(812),plot(t,z(:,2)*r2d, t,ybar(:,2)*r2d,'r--'); grid; ylabel('rdot (°/s2)');
subplot(813),plot(t,z(:,3),     t,ybar(:,3),    'r--'); grid; ylabel('ay (m/s2)');
subplot(814),plot(t,z(:,4)*r2d, t,ybar(:,4)*r2d,'r--'); grid; ylabel('p (°/s)');
subplot(815),plot(t,z(:,5)*r2d, t,ybar(:,5)*r2d,'r--'); grid; ylabel('r (°/s)');
subplot(816),plot(t,u(:,1)*r2d);                grid; ylabel('{\delta_a} (°)');
subplot(817),plot(t,u(:,2)*r2d);                grid; ylabel('{\delta_r}  (°)');
subplot(818),plot(t,u(:,3)    );                grid; ylabel('{vk}  (m/s)');...
xlabel('Time in sec');  

% Convergence plot: estimated parameter +- standard deviations versus iteration count
iterations = 0:numIters;
figure; zoom;
subplot(521);errorbar(iterations, thetaHist(1,:),thetaStd(1,:));ylabel('Lp');          xlabel('iteration #');grid
title('Convergence of parameter estimates with error bounds')
subplot(522);errorbar(iterations, thetaHist(6,:),thetaStd(6,:));ylabel('Np');          xlabel('iteration #');grid
subplot(523);errorbar(iterations, thetaHist(2,:),thetaStd(2,:));ylabel('Lr');          xlabel('iteration #');grid
subplot(524);errorbar(iterations, thetaHist(7,:),thetaStd(7,:));ylabel('NLr');         xlabel('iteration #');grid
subplot(525);errorbar(iterations, thetaHist(3,:),thetaStd(3,:));ylabel('L{\delta}a');  xlabel('iteration #');grid
subplot(526);errorbar(iterations, thetaHist(8,:),thetaStd(8,:));ylabel('NL{\delta}a'); xlabel('iteration #');grid
subplot(527);errorbar(iterations, thetaHist(4,:),thetaStd(4,:));ylabel('L{\delta}r');  xlabel('iteration #');grid
subplot(528);errorbar(iterations, thetaHist(9,:),thetaStd(9,:));ylabel('NL{\delta}r'); xlabel('iteration #');grid
subplot(529);errorbar(iterations, thetaHist(5,:),thetaStd(5,:));ylabel('Lv');          xlabel('iteration #');grid
subplot(5,2,10);errorbar(iterations, thetaHist(10,:),thetaStd(10,:));ylabel('Nv');     xlabel('iteration #');grid


