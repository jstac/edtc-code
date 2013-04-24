% Filename: kurtzbellman.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 5.1

% The MATLAB code for this listing is put in a class definition
% because MATLAB can only have one function or class definition
% per file.  By putting the code in a class definition, we can
% have all the code in one file.  At the bottom of the file is
% a test of the code which shows how to use the class.

classdef kurtzbellman < handle
    properties
        beta = 0.5;
        rho = 0.9;
        B = 10;
        M = 5;
        S;  % State space = 0,...,B + M
        Z;  % Shock space = 0,...,B
    end

    methods
        function self = kurtzbellman
            self.S = 0:(self.B + self.M);
            self.Z = 0:self.B;
        end
        
        function u = U(self,c)
            % Utility function
            u = c^self.beta;
        end

        function p = phi(self,z)
            % Probability mass function,uniform distribution
            if 0 <= z && z <= self.B
                p = 1.0 / length(self.Z);
            else
                p = 0;
            end
        end

        function g = Gamma(self,x)
            % The correspondence of feasible actions .
            g = 0:min(x,self.M);
        end

        function Tv = T(self,v)
            % An implementation of the Bellman operator.
            % Parameters: v is a sequence representing a
            % function defined on S. Returns: Tv,a list.
            Tv = zeros(1,length(self.S));
            for x = self.S
                % Compute the value of the objective function for each
                % a in Gamma (x), and store the result in vals
                vals = zeros(1,length(Gamma(self,x)));
                for a = Gamma(self,x)
                    Sum = 0;
                    for z = self.Z
                        Sum = Sum + v(a + z + 1) * self.phi(z);
                    end
                    y = U(self,(x - a)) + self.rho * Sum;
                    vals(a + 1) = y;
                end
                Tv(x + 1) = max(vals);
            end
        end
    end
end

% The following test shows how to use the class.  Put the code in a
% separate file, in the same directory as kurtzbellman.m
%
% kb = kurtzbellman;
% w = zeros(1, length(kb.S));
% err = 1;
% while err > 0.001
%     v = kb.T(w);
%     err = max(abs(v - w));
%     w = v;
%     hold on;
%     plot(w);
% end
% hold off;

