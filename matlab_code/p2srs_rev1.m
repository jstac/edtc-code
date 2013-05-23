% Filename: p2srs_rev1.m
% Author: Tomohito Okabe
% Date: March 2013
% Corresponds to: Listing 5.3


function F = p2srs_rev1(x, z, p)
    % Given a kernel p, takes p on S = {1,...,N},
    % and reurns F(x,z;p) which represents it as an SRS.
    % Returns : A value (NOT a function!) of F
    S = 1:length(p(1,:));
    a = 0;
        for y = S
            if a < z && z <= a + p(x,y)
                F = y;
                return;
            end
            a = a + p(x,y);
        end
end

    
% Example of usage (put this in a separate file in the same directory).
%
% p = [0.971,0.029,0.000;  
%      0.145,0.778,0.077;
%      0.000,0.508,0.492];
% x = 2;
% z = 0.7;
% F = p2srs_rev1(x, z, p)  % should return 2
