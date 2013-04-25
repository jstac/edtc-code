% Filename: polyclass.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 2.2

classdef polyclass
    properties
        coef;
    end
    
    methods
        function self = polyclass(coef)
            % Creates an instance p of the Polynomial class,
            % where p(x) = coef[0] x^0 + ... + coef[N] x^N.
            self.coef = coef;
        end
        
        function y = evaluate(self,x)
            % Reverse the order of coef, and then use MATLAB's 
            % polyval function to evaluate
            y = polyval(self.coef(end:-1:1),x);
        end
        
        function self = differentiate(self)
            len = length(self.coef);
            new_coef = zeros(1,len);
            for i = 1:len
                new_coef(i) = (i-1) * self.coef(i);  
            end
            % Remove the first element, which is zero
            new_coef(1) = [];
            % And reset coefficients data to new values
            self.coef = new_coef;
        end
    end
end
