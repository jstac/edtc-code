% Filename: cpdynam_rev1.m
% Author: Tomohito Okabe
% Date: April 2013
% Corresponds to: Listing 6.7

% Note: All of the code is wrapped in a class definition so that it can
% be put in one file.  See the comments in kurtzbellman.m.  An example
% of usage is given below.

global alpha a c
    alpha = 0.8;
    a = 5.0;
    c = 2.0;

 W = betarnd(5, 5, 1, 1000) * c + a; % Shock obs.
 P = @(x) 1.0/x;     % Inverse demand function
 D = P;
      

% The following test shows how to use the codes. lininterp_rev1.m must 
% be in the current folder.
 
% gridsize = 150;
% grid = linspace(a, 35, gridsize);
% tol = 0.0005;
% vals = zeros(1, gridsize);
% new_vals = zeros(1, gridsize);
% for i = 1:gridsize
%     vals(i) = P(grid(i));
% end
% while 1
%     hold on;
%     plot(grid, vals);
%     p = @(z)lininterp_rev1(grid, vals,z);
%     f = @(x) T(p, x, W, P, D);
%     for i = 1:gridsize
%         new_vals(i) = f(grid(i));
%     end
%     if max(abs(new_vals - vals)) < tol
%         break
%     end
%     vals = new_vals;
% end
% hold off;


