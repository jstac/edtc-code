% Filename: fix_point.m
% Author: Tomohito Okabe
% Date: April 2013
% Corresponds to: fixpoint in Listing 6.7

function fp = fix_point(h, lower, upper)
% Computes the fixed point of h on [upper,lower] using fzero, 
% which finds the  zeros (roots) of a univariate function.
% Inputs : h is a function and lower and upper are numbers 
% (floats or integers).

fp = fzero(@(x) x - h(x), [lower, upper]);

end