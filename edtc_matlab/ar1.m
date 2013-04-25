% Filename: ar1.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 8.1

a = 0.5;
b = 1;
X = zeros(1, 101);    % Create an empty array to store path
X(1) = normrnd(0, 1); % X_0 has dist N(0, 1)
for t = 1:100
    X(t + 1) = normrnd(a * X(t) + b, 1);
end
