% Filename: fvi.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 6.5

% Note: All of the code is wrapped in a class definition so that it can
% be put in one file.  See the comments in kurtzbellman.m.  An example
% of usage is given below.

classdef fvi < handle
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
        function self = fvi
            self.W = exp(randn(1,1000)); % Draws of shock
            self.grid = linspace(0,self.gridmax^(1e-1),self.gridsize).^10;
        end
        
        function u = U(self,c)      % Utility function
            u = 1 - exp(- self.theta * c);
        end
        
        function pro = f(self,k,z)  % Production function
            pro = (k^self.alpha) * z;
        end
        
        function m = maximum(self,h,a,b)
            m = h(fminbnd(@(x) -h(x),a,b));
        end
        
        function li = bellman(self,w)
            % The approximate Bellman operator.
            % Parameters : w is an instance of lininterp (see the file
            %              lininterp.m)
            % Returns : An instance of lininterp
            
            len = length(self.grid);
            vals = zeros(1,len);
            for i = 1:len
                y = self.grid(i);
                h = @(k) U(self,(y - k)) +... 
                self.rho * mean(w.interp(f(self,k,self.W)));
                vals(i) = self.maximum(h,0,y);
            end
            li = lininterp(self.grid,vals);
        end
    end
end

% The following test shows how to use the class.  Put the code in a
% separate file, in the same directory as fvi.m and lininterp
%
% f = fvi;
% v = lininterp(f.grid, f.U(f.grid));
% 
% for i = 1:10
%     hold on;
%     plot(f.grid, v.Y);
%     w = f.bellman(v);
%     v = w;
% end
% hold off;
