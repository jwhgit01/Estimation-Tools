close all
clear
clc
addpath(genpath('../'))

% Generate truth data
rng('default')
N = 100;
alpha = 1;
beta = 3;
gamma = 2;
R = 0.01;
x = sort(-2*log(rand(N,1)));
y = alpha*ones(N,1) + beta*ones(N,1).*exp(-gamma*x);
z = y + sqrt(R)*randn(N,1);

% Create the Gauss-Newton estimator
GN = gaussNewtonEstimator(@measurementModel_GN,R)

% Initial guess
theta0 = 4*rand(3,1);

% Perform batch nonlinear least squares
[thetahat,Jopt,P,termflag] = GN.estimate(z,x,theta0,[])

% Estimated output
yhat = measurementModel_GN([],thetahat,x,[]);

% Plot
figure
hold on
plot(x,y,'LineWidth',1.5)
plot(x,z,':','LineWidth',1.5)
plot(x,yhat,'--','LineWidth',1.5)
hold off
legend('y','z','yhat')
