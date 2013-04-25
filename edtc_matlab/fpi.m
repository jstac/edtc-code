% Filename: fpi.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 6.6

% Note: All of the code is wrapped in a class definition so that it can
% be put in one file.  See the comments in kurtzbellman.m.  An example
% of usage is given below.

classdef fpi < handle
    properties
        theta = 0.5;
        alpha = 0.8;
        rho = 0.9;
        W;
        gridmax = 8;
        gridsize = 150;
        grid;
    end
    
    methods
        function self = fpi
            self.W = exp(randn(1, 1000)); % Draws of shock
            self.grid = linspace(0, self.gridmax^(1e-1), self.gridsize).^10;
        end
        
        function u = U(self, c)
        % Utility function
            u = 1 - exp(- self.theta * c);
        end
        
        function pro = f(self, k, z)
        % Production function
            pro = (k^self.alpha) * z;
        end
        
        function max = maximizer(self, h, a, b)
            max = fminbnd(@(x) -h(x), a, b);
        end
        
        function t = T(self, sigma, w)
        % Implements the operator L T_sigma
        % Parameters: sigma and w should be instances of lininterp
            len = self.gridsize;
            vals = zeros(1, len);
            for i = 1:len
                x = self.grid(i);
                Tw_y = self.U(x - sigma.interp(x)) + ... 
                self.rho * mean(w.interp(self.f(sigma.interp(x), self.W)));
                vals(i) = Tw_y; 
            end
            t = lininterp(self.grid, vals);
        end
        
        function g = get_greedy(self, w)
        % Computes a w-greedy policy, where w is an instance of lininterp
            len = self.gridsize;
            vals = zeros(1, len);
            for i = 1:len
                x = self.grid(i);
                h = @(k) self.U(x - k) + ...
                self.rho * mean(w.interp(self.f(k, self.W)));
                vals(i) = self.maximizer(h, 0, x);
            end
            g = lininterp(self.grid, vals);
        end
        
        function v = get_value(self, sigma, v)
        % Computes an approximation to v_sigma, the value of following 
        % policy sigma. Function v is a guess of v_sigma. 
        % Parameters: sigma and v are both instances of lininterp
            tol = 1e-2; % Error tolerance
            while 1
                new_v = self.T(sigma, v);
                err = max(abs(new_v.Y - v.Y));
                if err < tol
                    v = new_v;
                    return;
                end
                v = new_v;
            end
        end
    end
end

% The following test shows how to use the class.  Put the code in a
% separate file, in the same directory as fpi.m and lininterp
%
% tol = 0.005;
% f = fpi;
% % Choose a guess of the optimal policy
% sigma = f.get_greedy(lininterp(f.grid, f.U(f.grid)));
% Compute v_sigma using U as a starting point for the interative
% procedure in get_value
% v = f.get_value(sigma, lininterp(f.grid, f.U(f.grid)));
% while 1
%     hold on;
%     plot(f.grid, sigma.Y);
%     sigma_new = f.get_greedy(v);
%     % Compute v_sigma_new using v as a starting point for the iteration
%     v_new = f.get_value(sigma_new, v);
%     err = max(abs(v_new.Y - v.Y));
%     if err < tol
%         break
%     else
%         v = v_new;
%         sigma = sigma_new;
%     end
% end
% hold off;

