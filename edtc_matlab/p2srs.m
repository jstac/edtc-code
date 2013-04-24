% Filename: p2srs.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 5.3

% Note: The code below gives an example of usage

function r = p2srs(p)
    % Takes a kernel p on S = {1,...,N} and returns a function 
    % F(x,z) which represents it as an SRS.
    % Parameters : p is a matrix
    % Returns : A function F with arguments (x,z).
    S = 1:length(p(1,:));
    function f = F(x,z)
        a = 0;
        for y = S
            if a < z && z <= a + p(x,y)
                f = y;
                return;
            end
            a = a + p(x,y);
        end
    end
    r = @F;
end

% Example of usage (put this in a separate file in the same directory).
%
%
% pH = [0.971,0.029,0.000;  
%       0.145,0.778,0.077;
%       0.000,0.508,0.492];
% F = p2srs(pH);
% F(2, 0.7)  % should return 2
