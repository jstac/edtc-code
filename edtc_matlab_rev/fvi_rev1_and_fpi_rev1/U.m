% Filename: U.m
% Author: Tomohito Okabe
% Date: April 2013
% Corresponds to: U(c) in Listing 6.5 and 6.6

function [ utility ] = U(c)

global theta

utility = 1 - exp(-theta.* c);      % Utility function

end

