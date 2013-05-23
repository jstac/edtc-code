% Filename: quadmap1.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 4.1

datapoints = zeros(1,200);
x = 0.11;
for t = 1:200
    datapoints(t) = x;
    x = 4 * x * (1-x);
end
plot(datapoints)
