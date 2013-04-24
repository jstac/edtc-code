% Filename: testds.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 4.3

quadmap = @(x)4 * x * (1 - x);
q = ds(quadmap,0.1);          % Create an instance q of ds
T1 = q.trajectory(100);       % T1 holds trajectory from 0.1

q.x = 0.2;                    % Reset current state to 0.2
T2 = q.trajectory(100);       % T2 holds trajectory from 0.2
