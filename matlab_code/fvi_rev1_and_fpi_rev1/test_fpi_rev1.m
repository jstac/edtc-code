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

sigma = get_greedy(grid, @(x) lininterp_rev1(grid, U(grid),x), W);

[new_v, new_v_func] = get_value(grid, W, sigma, @(x) lininterp_rev1(grid, U(grid), x));

v = zeros(1, len);
v_func = new_v_func;

tol = 0.005;
err = 1;

while err > tol
     hold on;
     plot(grid, sigma);
     new_sigma = get_greedy(grid, v_func, W);
     [new_v, new_v_func] = get_value(grid, W, new_sigma, v_func);
     err = max(abs(new_v - v));     
     v = new_v;
     v_func = new_v_func;
     sigma = new_sigma;
end
