% Filename: testgenfinitemc_rev1.m
% Author: Tomohito Okabe
% Date: February 2013
% Corresponds to: Listing 4.5

clear all
close all

N = 1000;                             % Number of periods

pH = [0.971,0.029,0.000;              % Stochastic kernel
      0.145,0.778,0.077;
      0.000,0.508,0.492];

psi = [0.3,0.4,0.3];                  % Initial condition
len = length(psi);
rs = randsample(len,1,true,psi);      % Use randsample fuction returns i with
                                      % probability phi[i], where phi serves 
                                      % as a vector of weights.
T1 = genfinitemc_rev1(pH,rs,N,len);   % T1 holds Markov-(pH,psi)


psi2 = [0.8,0.1,0.1];                 % Alternative initial cond.
len2 = length(psi2);
rs2 = randsample(len,1,true,psi2);
T2 = genfinitemc_rev1(pH,rs2,N,len2); % T2 holds Markov-(pH,psi2)

