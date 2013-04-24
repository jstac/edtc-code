% Filename: kurtzvsigma.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 5.2

% Note: See the comments below for an example of usage.

function v_sigma = kurtzvsigma(kb,sigma)
    % Computes the value of following policy sigma.  Here kb
    % is an instance of the class kurtzbellman, as defined in
    % kurtzbellman.m
    
    % Set up the stochastic kernel p_sigma as a 2D array :
    len = length(kb.S);
    p_sigma = zeros(len,len);
    for x = kb.S
        for y = kb.S
            p_sigma(x + 1,y + 1) = phi(kb,(y - sigma(x + 1)));
        end
    end
    % Create the right Markov operator M_sigma :
    M_sigma = @(h)p_sigma * h;
    
    % Set up the function r_sigma as an array :
    % Initialize r_sigma into a column vector
    r_sigma = zeros(len,1); 
    for x = kb.S
        r_sigma(x + 1) = U(kb,(x - sigma(x + 1)));
    end
    v_sigma = zeros(len,1);
    discount = 1;
    for i = 1:50
        v_sigma = v_sigma + discount * r_sigma;
        r_sigma = M_sigma(r_sigma);
        discount = discount * kb.rho;
    end
end

% Example of usage (put this in a separate file in the same 
% directory as kurtzbellman.m and kurtzvsigma.m
%
% BO = kurtzbellman;
% m = ones(1, 16);
% kurtzvsigma(BO, m)

