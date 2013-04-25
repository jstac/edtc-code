% Filename: testsrs.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 6.2

alpha = 0.5;
sigma = 0.2;
s = 0.5;
delta = 0.1;
% Define F(k, z) = s * k ^ alpha * z + (1 - delta ) * k
F = @(k,z) s * (k^alpha) * z + (1 - delta) * k;
lognorm = @() lognrnd(0,sigma);

solow_srs = srs(F,lognorm,1.0);
P1 = sample_path(solow_srs,500); % Generate Path from X = 1
solow_srs.X = 60;                % Reset the current state
P2 = sample_path(solow_srs,500); % Generate Path from X = 60
