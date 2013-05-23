% Filename: lininterp_rev1.m
% Author: Tomohito Okabe
% Date: April 2013
% Corresponds to: Listing 6.4

function g = lininterp_rev1(X,Y,z)
            % Uses MATLAB's interp1 function for interpolation.
            % The z values are truncated so that they lie inside
            % the grid points.  The effect is that evaluation of
            % a point to the left of X(1) returns Y(1), while 
            % evaluation of a point to the right of X(end) returns
            % Y(end)
          
            z = max(z, X(1));
            z = min(z, X(end));
            g = interp1(X, Y, z);
end
