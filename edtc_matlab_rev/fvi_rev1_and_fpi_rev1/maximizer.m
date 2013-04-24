% Filename: maximizer.m
% Author: Tomohito Okabe
% Date: April 2013
% Corresponds to: maximizer(h,a,b) in Listing 6.6

function [max] = maximizer(h, a, b)

% Inputs : a function handle, h; constants, a and b
% Return : a maximizer for h in domain (a,b)

max = fminbnd(@(k) -h(k), a, b);  

end
        