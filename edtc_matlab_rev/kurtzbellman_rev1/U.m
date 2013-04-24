% Filename: U.m
% Author: Tomohito Okabe
% Date: February 2013
% Corresponds to: U(c) in Listing 5.1

function [ utility ] = U( c )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
global beta

utility = c.^beta;

end

