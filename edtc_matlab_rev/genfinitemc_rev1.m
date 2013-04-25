% Filename: genfinitemc_rev1.m
% Author: Tomohito Okabe
% Date: February 2013
% Corresponds to: Listing 4.4

function [ path ] = genfinitemc_rev1(p, X, n, len)
% Inputs:
% p -> stochastic kernel; 
% x -> the initial state;
% n -> # of periods;
% len -> # of states;
%
% Output:
% path -> series of the finite Markov chain

% Set the initial state
path(1) = X;

% Generate a sample path of length n, starting at the current state
for i = 1:n-1
    psi = p(path(i),:);
    path(i+1) = randsample(len,1,true,psi);   
end

end

