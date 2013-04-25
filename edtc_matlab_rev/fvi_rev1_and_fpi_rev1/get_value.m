% Filename: get_value.m
% Author: Tomohito Okabe
% Date: April 2013
% Corresponds to: get_value(sigma,v) in Listing 6.6

function  [new_v, new_v_func] = get_value(grid, W, sigma, v_func)

% Computes an approximation to v_sigma, the value of following 
% policy sigma. Function v is a guess of v_sigma.

% Inputs : A grid array, grid;
%          A shock array, W;
%          A policy array, sigma;          
%          An instance of lininterp (see 
%          the file lininterp_rev1.m), v_func (a function handle).
%
% Returns : An output array for v, new_v; 
%           An instance of lininterp, new_v_func (a function handle).

global gridsize

tol = 1e-2; % Error tolerance
err = 1;

v = zeros(1,gridsize);

while err > tol
    [new_v, new_v_func] = T(grid, W, sigma, v_func);
    err = max(abs(new_v - v));
    v_func = new_v_func;
    v = new_v;
end
