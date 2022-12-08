function Xknew = resampleDiracDelta(Xk,wk)
%resampleDiracDelta
%
% Copyright (c) 2022 Jeremy W. Hopwood. All rights reserved.
%
% This function performs resampling of particles in a particle filter by
% sampling from the approximate a posteriori state vector proposal
% distribution
%
%                            Ns
%               p[x(k)|Zk] = SUM wi(k) dirac(x(k) - Xi(k))          (1)
%                            i=1
%
% This function uses the "Alternate Re-sampling Algorithm" covered the
% lectures of the Virginia Tech course AOE 5784 - Estimation and
% Filtering, taught by Prof. Mark L. Psiaki in the Fall 2022 semester. 
%
% Inputs:
%
%   Xk      The nx x Ns matrix of particle state vectors at sample k, where
%           Ns is the number of particles.
%
%   wk      The Ns x 1 vector of particle weights at sample k.
%  
% Outputs:
%
%   Xknew    The nx x Ns matrix of particles resampled from (1)
%

% Get the number of particles and initialize the output
Ns = size(Xk,2);
Xknew = Xk;

% The procedure for generating random samples from this distribution starts
% by defining the following constants.
epsil = 1e-10;
c = zeros(Ns+1,1);
for ii = 2:Ns
    c(ii,1) = sum(wk(1:ii-1,1));
end
c(Ns+1,1) = 1 + epsil;

% Sample eta1 from a uniform distribution.
eta1 = rand/Ns;

% Given these constants, the following algorithm samples Ns new samples
% from the approximate a posteriori state-vector distribution that has been
% defined in (1). Set i = 1 and l = 1 and begin iteratively sampling.
ii = 1;
ll = 1;
while ll <= Ns
    
    % Set etal and comapre to computed constants. If it is below the
    % threshold, set the lth particle to have the valuenof the old ith
    % particle. Then, move on to sample the next particle.
    etal = eta1 + (ll-1)/Ns;
    if etal < c(ii+1)
        Xknew(:,ll) = Xk(:,ii);
        ll = ll + 1;
        continue
    end

    % If the ith particle was not good, check the next one.
    ii = ii + 1;

end

end