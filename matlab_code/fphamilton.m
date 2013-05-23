% Filename: fphamilton.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 4.6

pH = [0.971,0.029,0.000;  % Hamilton's kernel
      0.145,0.778,0.077;
      0.000,0.508,0.492];
I = eye(3);               % 3 by 3 identity matrix
Q = ones(3);              % Matrix of ones
b = ones(3,1);            % Vector of ones
A = transpose(I - pH + Q);
solution = A \ b;
