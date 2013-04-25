% Filename: test_fvi_rev1.m
% Author: Tomohito Okabe
% Date: April 2013

clear all
close all

global theta alpha rho gridmax gridsize
    theta = 0.5;
    alpha = 0.8;
    rho = 0.9;
    gridmax = 8;
    gridsize = 150;
    
    
W = exp(randn(1,1000));  % Draws of shock

grid = linspace(0,gridmax^(1e-1),gridsize).^10; % Grid for state space

len = length(grid);

vals = zeros(1,len);

v_func = @(x) lininterp_rev1(grid, U(grid),x); % Initial input


 for i = 1:10
     [w, w_func] = bellman(grid,v_func,W);
     v_func = w_func;
     hold on;
     plot(grid, w);
 end
 
hold off;
   