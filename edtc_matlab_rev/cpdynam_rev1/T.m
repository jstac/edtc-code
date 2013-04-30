% Filename: T.m
% Author: Tomohito Okabe
% Date: April 2013
% Corresponds to: T in Listing 6.7


function t = T(p, x, W, P, D)

global alpha

% Computes Tp(x), where T is the pricing functional operator.
% Parameters : p is an instance of lininterp and x is a number.

y = alpha * mean(p(W));
            if y <= P(x)
                t = P(x);
                return;
            end
h = @(r) alpha * mean(p(alpha * (x - D(r)) + W));
t = fix_point(h, P(x), y);
end

