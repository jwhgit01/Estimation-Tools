addpath('../Models')
addpath('../../src/dynamics')


%% Discrete-Time Linear Stationary System
close all
clear
clc

N = 100;
nx = 2;
nu = 1;
nv = 1;
nz = 1;

F = [0.8, 0.08; -3.5, 0.7];
G = [0; 1];
Gam = [0.004; 0.09];
H = [2, 0.3];

Q = 4;
R = 0.01;

x0 = [-1; 1];

v = chol(Q)*randn(N,nv);
w = chol(R)*randn(N,nz);

K = (0:N).';
x = zeros(N+1,nx);
x(1,:) = x0.';
y = zeros(N,nz);
u = sin(0.2*K);

for k = 0:N-1
    idx = k + 1;
    xk = x(idx,:).';
    uk = u(idx,:).';
    vk = v(idx,:).';
    xkp1 = F*xk + G*uk + Gam*vk;
    ykp1 = H*xkp1;
    x(idx+1,:) = xkp1.';
    y(k+1,:) = ykp1.';
end
z = y + w;

vars = {'F','G','Gam','H','Q','R','x','y','z','u','v','w'};
save('DiscreteTimeLinearStationarySystem.mat',vars{:})

%% Discrete-Time Linear Non-Stationary System
close all
clear
clc

N = 100;
nx = 2;
nu = 1;
nv = 1;
nz = 1;

F = @(k) [1+0.8*cos(0.4*k), 0.08; -3.5, -2*exp(-0.1*k)];
G = [0; 1];
Gam = @(k) [0.004*sin(0.1*k); 0.09];
H = [2, 0.3];

Q = 4;
R = 0.01;

x0 = [-1; 1];

v = chol(Q)*randn(N,nv);
w = chol(R)*randn(N,nz);

K = (0:N).';
x = zeros(N+1,nx);
x(1,:) = x0.';
y = zeros(N,nz);
u = sin(0.2*K);

for k = 0:N-1
    idx = k + 1;
    xk = x(idx,:).';
    uk = u(idx,:).';
    vk = v(idx,:).';
    xkp1 = F(k)*xk + G*uk + Gam(k)*vk;
    ykp1 = H*xkp1;
    x(idx+1,:) = xkp1.';
    y(k+1,:) = ykp1.';
end
z = y + w;

vars = {'F','G','Gam','H','Q','R','x','y','z','u','v','w'};
save('DiscreteTimeLinearNonstationarySystem.mat',vars{:})

%% Continuous-Time Linear Time-Invariant System
close all
clear
clc

T = 10;
dt = 0.1;
nx = 2;
nu = 1;
nv = 1;
nz = 1;

A = [-0.3, 0.9; -4.1, -0.15];
B = [0; 1];
Del = [0.04; 0.9];
C = [2, 0.3];

Q = 4;
R = 0.01;

x0 = [-1; 1];

t = (0:dt:T).';
N = length(t)-1;
w = chol(R)*randn(N,nz);

% Assume piecewise constant controls
u = sin(pi*t);
ufun = @(tau) interp1(t,u,tau,'previous');

% Use 10x the sampling rate to approximate continuous-time white noise
dtv = dt/10;
tEM = (0:dtv:T-dtv).';
Nv = length(tEM);
vtrue = chol(Q)*randn(Nv,nv);
wtilfun = @(t) interp1(tEM,vtrue,t,'previous');
vsampled = wtilfun(t);

odefun = @(t,x) A*x + B*ufun(t) + Del*wtilfun(t);
[t,x] = ode45(odefun,t,x0);

for k = 1:N
    xk = x(k+1,:).';
    yk = C*xk;
    y(k,:) = yk.';
end
z = y + w;

vars = {'A','B','Del','C','Q','R','x','y','z','u','tv','vtrue','vsampled','w'};
save('ContinuousTimeLinearTimeInvariantSystem.mat',vars{:})

figure
hold on
plot(t(2:end),y)
plot(t(2:end),z)
hold off

figure
plot(t,x)

figure
plot(t(2:end),w)

figure
hold on
plot(tEM,vtrue)
plot(t,vsampled)
hold off

