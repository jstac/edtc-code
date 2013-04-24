% Filename: testmc.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 4.5

pH = [0.971,0.029,0.000;
      0.145,0.778,0.077;
      0.000,0.508,0.492];

psi = [0.3,0.4,0.3];                  % Initial condition
len = length(psi);
rs = randsample(1:len,1,true,psi);    % Use randsample fuction returns i with
                                      % probability phi[i], where phi is a
                                      % sequence
h = mc(pH,rs);                        % Create an instance of class mc
T1 = sample_path(h,1000);             % Series is Markov -(p, psi)

psi2 = [0.8,0.1,0.1];                 % Alternative initial cond.
len2 = length(psi2);
h.X = randsample(1:len2,1,true,psi2); % Reset the current state
T2 = sample_path(h,1000);             % Series is Markov -(p, psi2)
