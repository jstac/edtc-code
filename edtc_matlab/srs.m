% Filename: srs.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 6.1

classdef srs < handle
    properties
        % Parameters : F and phi are functions, where
        % phi() returns a draw from phi. X is a
        % number representing the initial condition
        F;
        phi;
        X;
    end
    
    methods
        function self = srs(F,phi,X)
            % Represents X_{t + 1} = F(X_t,W_{t + 1}); W ~ phi
            self.F = F;
            self.phi = phi;
            self.X = X;
        end
        
        function update(self)
            % Update the state according to X = F(X, W)
            self.X = self.F(self.X,self.phi());
        end
        
        function path = sample_path(self,n)
            % Generate path of length n from current state
            path = zeros(1,n);
            for i = 1:n
                path(i) = self.X;
                self.update;
            end
        end
    end
end
