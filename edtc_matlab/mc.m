% Filename: mc.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 4.4

classdef mc < handle
    
    properties
        % p[x] is a sequence
        % of length N for each x, and represents p(x,dy).
        % The parameter X is an integer in S.
        p;
        X;
    end

    methods
        function self = mc(p,X)
            % Create an instance with stochastic kernel
            % p and current state X.Here p[x] is an array of length N
            % for each x, and represents p(x,dy).
            % The parameter X is an integer in S.
            self.p = p;
            self.X = X;
        end

        function update(self)
            % Update the state by drawing from p(X,dy).
            px = self.p((self.X),:);
            len = length(px);
            self.X = randsample(1:len,1,true,px);
        end

        function path = sample_path(self,n)
            % Generate a sample path of length n, starting
            % from the current state .
            path = zeros(1,n);
            for i = 1:n
                path(i) = self.X;
                self.update;
            end
        end
    end
end
