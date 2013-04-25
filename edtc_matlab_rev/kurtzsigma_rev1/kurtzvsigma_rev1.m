% Filename: kurtzvsigma_rev1.m
% Author: Tomohito Okabe
% Date: February 2013
% Corresponds to: Listing 5.2

% Note: See the comments below for an example of usage.

function v_sigma = kurtzvsigma_rev1(sigma)
    % Computes the value of following policy sigma.
    
  
    % Copy of kurtzbellman_rev1 
    global beta rho B M 
    beta = 0.5;
    rho = 0.9;
    B = 10;
    M = 5;
    S = [0:B+M];  % State space = 0,...,B + M
    Z = [0:B];    % Shock space = 0,...,B
    

    
    % Set up the stochastic kernel p_sigma as a 2D array :
    len = length(S);
    p_sigma = zeros(len,len);
    y = S;
    for x = S
        p_sigma(x+1,:) = phi(y - sigma(x+1));
    end
    
    % Create the right Markov operator M_sigma :
    
    % Set up the function r_sigma as an array :
    % Initialize r_sigma into a column vector
    
    r_sigma = zeros(len,1);
    for x = S
        c = x-sigma(x+1);
        if c <0
            c =0;
        end
        r_sigma(x+1) = U(c);
    end
      
    v_sigma = zeros(len,1);
    discount = 1;
    for i=1:50
        v_sigma = v_sigma + discount * r_sigma;
        M_sigma = p_sigma * r_sigma;
        r_sigma = M_sigma;
        discount = discount * rho;
    end

 % Example of usage (put this in a separate file in the same 
 % directory as kutzvsigma_rev1.m / U.m / phi.m 
 %
 % clear all
 % close all
 %
 % sigma = ones(1,16);
 % kurtzvsigma_rev1(sigma)