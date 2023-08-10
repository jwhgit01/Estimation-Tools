close all
clear
clc
addpath(genpath('../'))
load ContinuousTimeNonlinearSystem.mat

xhat0 = [0;0;0];
P0 = 10*eye(3);

[xhat,P,nu,epsnu,sigdig] = extendedKalmanFilterCD(t,z,[],...
                       @nonlindyn_CT,@measmodel_CT,Q,R,xhat0,P0,50,params);

figure
hold on
plot(t,x)
plot(t,xhat,'--')
hold off
grid on
