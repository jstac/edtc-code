% Filename: get_greedy.m
% Author: Tomohito Okabe
% Date: April 2013
% Corresponds to: get_greedy(w) in Listing 6.6

function  [new_w] = get_greedy(grid,w_func,W)

global rho gridsize
 % Computes a w-greedy policy, where w_func is an instance of lininterp_rev1

len = gridsize;
vals = zeros(1, len);

for i = 1:len
    x = grid(i);
    h = @(k) U(x - k) + rho * mean(w_func(f(k, W)));
    vals(i) = maximizer(h, 0, x);
end
new_w = vals;
