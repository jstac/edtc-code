% Filename: T.m
% Author: Tomohito Okabe
% Date: February 2013
% Corresponds to: T(v) in Listing 5.1

function [ Tv ] = T(v)

% Set parameters
global beta rho B M 
beta = 0.5;
rho = 0.9;
B = 10;
M = 5;

S = [0:B+M];  % State space = 0,...,B + M
Z = [0:B];    % Shock space = 0,...,B

% Call phi.m to define the probability mass function, uniform distribution.
p = phi(Z);   

% Call Gamma.m to compute the correspondence of feasible action
G = Gamma(S); 

C = ones(1,1+M)'*S - G; % Generate a grid for consumption
C = max(0,C);           % Replace negative values with zero
u = U(C);               % Call U.m to compute utility

% An implementation of the Bellman operator
% Parameter: v is a sequence representing a function on S.
% Returns: Tv 
for i= 1:size(G,1)      % Compute the value of the objective junc for each
    for j = 1:length(S) % a in G, and stores the result in y    
        a = G(i,j);    
        v_new(i,j) = sum(v(a+1:a+length(Z)).*p);
    end 
end

y = u + rho*v_new;

for i=1:length(S)
    Tv(i) = max(y(:,i));
end


end

% The following test shows how to use the class.  Put the code in a
% separate file, in the same directory as kurtzbellman_rev1.m
%
% clear all
% close all
%
% w = zeros(1, 16); % 16 = B + M
% err = 1;
% while err > 0.001
%    v = kurtzbellman_rev1(w);
%    err = max(abs(v - w));
%    w = v;
%    hold on;
%    plot(w);
% end
% hold off;