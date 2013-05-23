% Filename: ds_rev1.m
% Author: Tomohito Okabe
% Date: February 2013
% Corresponds to: Listing 4.2


function [ X ] = ds_rev1(h,x,n)
% Inputs:
% h -> an arbitrary handle function; 
% x   -> the initial state;
% n   -> size of length;
%
% Output:
% X -> trajectory

% Set the initial state
X(1)= x;

% Generate a tragectory of length n, starting at the current state
for i = 1 : n-1
    X(i+1) = h(X(i)); % Update the state of the system by applying h 
end

end

