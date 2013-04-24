% Filename: Gamma.m
% Author: Tomohito Okabe
% Date: February 2013
% Corresponds to: Gamma(x) in Listing 5.1

function [ G ] = Gamma(S)

% The correspondence of feasible actions = a (M+1) x (B+M+1) matrix
% Gamma = [Gamma(0), Gamma(1), ... Gamma(B+M)] 
% where Gamma(s) (s in S) is a (M+1)-column vector s.t.
% Gamma(s) = [0,1,...,M]'         if s >= M
% Gamma(s) = [0,1,...,m*,...,m*]' otherwise
% m* denotes the maximum value of the feasible action in the set. 
% The latter equation is for a matter of convenience to generate a grid for
% consumption.

global M

e = ones(1,length(S));
Gamma = [0:M]'*e;  
for i = 1:M-1       
for j = 1:M+1
    if j > i
       Gamma(j,i)= i-1;
    end
end
end

G = Gamma;

end

