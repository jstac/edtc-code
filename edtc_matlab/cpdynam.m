% Filename: cpdynam.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 6.7

% Note: All of the code is wrapped in a class definition so that it can
% be put in one file.  See the comments in kurtzbellman.m.  An example
% of usage is given below.

classdef cpdynam < handle
    properties
        alpha = 0.8;
        a = 5.0;
        c = 2.0;
        W;
        D;
        P;
    end
    
    methods
        function self = cpdynam
            self.W = betarnd(5, 5, 1, 1000) * self.c + self.a; % Shock obs.
            self.D = @(x) 1.0/x; % Demand curve
            self.P = self.D;     % Inverse of 1/x is just 1/x
        end

        function fp = fix_point(self, h, lower, upper)
            % Computes the fixed point of h on [upper,lower] using fzero, 
            % which finds the  zeros (roots) of a univariate function.
            % Parameters : h is a function and lower and upper are numbers 
            % (floats or integers).
            fp = fzero(@(x) x - h(x), [lower, upper]);
        end
        
        function t = T(self, p, x)
            % Computes Tp(x), where T is the pricing functional operator.
            % Parameters : p is an instance of lininterp and x is a number.
            y = self.alpha * mean(p.interp(self.W));
            if y <= self.P(x)
                t = self.P(x);
                return;
            end
            h = @(r) self.alpha * +...
                mean(p.interp(self.alpha * (x - self.D(r)) + self.W));
            t = self.fix_point(h, self.P(x), y);
        end
    end
end

% The following test shows how to use the class.  Put the code in a
% separate file, in the same directory as cpdynam.m and lininterp
%
% cp = cpdynam;
% gridsize = 150;
% grid = linspace(cp.a, 35, gridsize);
% tol = 0.0005;
% vals = zeros(1, gridsize);
% new_vals = zeros(1, gridsize);
% for i = 1:gridsize
%     vals(i) = cp.P(grid(i));
% end
% while 1
%     hold on;
%     plot(grid, vals);
%     p = lininterp(grid, vals);
%     f = @(x) cp.T(p, x);
%     for i = 1:gridsize
%         new_vals(i) = f(grid(i));
%     end
%     if max(abs(new_vals - vals)) < tol
%         break
%     end
%     vals = new_vals;
% end
% hold off;

