% Filename: f.m
% Author: Tomohito Okabe
% Date: April 2013
% Corresponds to: f(k,z) in Listing 6.5 and 6.6 

function [ prod ] = f(k, z)

global alpha

prod = (k.^alpha) .* z;          % Production function
        
end

