% Filename: lininterp.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 6.4

classdef lininterp
    % Provides linear interpolation in one dimension
    properties
        % Parameters : X and Y are sequences or arrays
        % containing the (x,y) interpolation points
        X;
        Y;
    end
    methods
        function self = lininterp(x,y)
            self.X = x;
            self.Y = y;
        end
        
        function i = interp(self,z)
            % Uses MATLAB's interp1 function for interpolation.
            % The z values are truncated so that they lie inside
            % the grid points.  The effect is that evaluation of
            % a point to the left of X(1) returns Y(1), while 
            % evaluation of a point to the right of X(end) returns
            % Y(end)
            z = max(z, self.X(1));
            z = min(z, self.X(end));
            i = interp1(self.X, self.Y, z);
        end
    end
end
