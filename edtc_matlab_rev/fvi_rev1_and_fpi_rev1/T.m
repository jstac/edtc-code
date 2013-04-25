% Filename: T.m
% Author: Tomohito Okabe
% Date: April 2013
% Corresponds to: T(sigma,w) in Listing 6.6

function  [t, t_func] = T(grid, W, sigma, w_func)

global rho gridsize

% Implements the operator L T_sigma

% Inputs: A grid array, grid;
%         A shock array, W;        
%         A policy array, sigma;
%         An instance of lininterp (see 
%         the file lininterp_rev1.m), w_func (a function handle);
% 
% Returns:  An output array for LT_sigma(w_func), t; 
%           An instance of lininterp, t_func (a function handle).

len = gridsize;
vals = zeros(1, len);

for i = 1:len
    x = grid(i);
    Tw_y = U(x - sigma(i)) + rho * mean(w_func(f(sigma(i), W)));
    vals(i) = Tw_y; 
end
t =  vals;
t_func = @(x) lininterp_rev1 (grid, vals, x);
end

