% Filename: testsrs_rev1.m
% Author: Tomohito Okabe
% Date: March 2013
% Corresponds to: Listing 6.2

clear all
close all

% Set parameters
global alpha sigma s delta
alpha = 0.5;
sigma = 0.2;
s = 0.5;
delta = 0.1;

% Define F(k, z) = s * k ^ alpha * z + (1 - delta ) * k
F = @(k,z) s * (k^alpha) * z + (1 - delta) * k;
lognorm = @() lognrnd(0,sigma);
X = 1;      
T = 500;

P1 = sample_path(F, lognorm, X, T); % Generate path from X = 1

X = 60;                             % Reset the current state
P2 = sample_path(F, lognorm, X, T); % Generate path from X = 60


% The below are commands to plot the paths as figure 6.1
% figure(1)
%    plot(P1);
%    hold on;
%    plot(P2);