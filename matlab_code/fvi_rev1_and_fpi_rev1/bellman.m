% Filename: bellman.m
% Author: Tomohito Okabe
% Date: April 2013
% Corresponds to: bellman(w) in Listing 6.5

function [b, b_func] = bellman (grid, w_func, W)

% The approximate Bellman operator.
% Inputs : A grid array, grid;
%          An instance of lininterp (see 
%          the file lininterp_rev1.m), w_func (a function handle);
%          A shock array, W.
%
% Returns : An output array for T(w_func), b; 
%           An instance of lininterp, b_func (a function handle).


global rho gridsize

len = gridsize;

 for i = 1:len
    y = grid(i);
    h = @(k) U(y - k) + rho * mean(w_func(f(k,W)));
    vals(i) = maximum(h,0,y);
 end
 
 b = vals;
 b_func = @(x) lininterp_rev1 (grid, vals, x);
