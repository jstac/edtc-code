% Filename: phi.m
% Author: Tomohito Okabe
% Date: February 2013
% Corresponds to: phi(z) in Listing 5.1

function [ dist ] = phi(Z)

global B

% Generate Z1 replacing zero elements with one, 
% and more-than-B elements with zero for the subsequent line. 
for i = 1:length(Z)
    if Z(i) == 0
        Z(i) =1;
    elseif Z(i)> B
        Z(i)=0;
    end
end

% Probability mass function, uniform distribution.
% length Z(Z>0)) counts the number of positive intergers in Z. 
dist = unidpdf(Z,length(Z(Z>0))); 

end

