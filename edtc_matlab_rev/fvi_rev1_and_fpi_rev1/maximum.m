% Filename: maximum.m
% Author: Tomohito Okabe
% Date: April 2013
% Corresponds to: maximum(h,a,b) in Listing 6.5 and 6.6

function [ m ] = maximum( h, a, b)

m = h(fminbnd(@(k) -h(k),a,b));

end

