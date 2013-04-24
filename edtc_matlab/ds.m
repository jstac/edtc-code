% Filename: ds.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing 4.2

classdef ds < handle

    properties
        % h is a function and x is a number
        % in S representing the current state .
        h;
        x;
    end
    
    methods
        function self = ds(h,x)
            self.h = h;
            self.x = x;
        end
        
        function update(self)
            % Update the state of the system by applying h.
            self.x = self.h(self.x);
        end
        
        function traj = trajectory(self,n)
            % Generate a trajectory of length n, starting 
            % at the current state .
            traj = zeros(1,n);
            for i = 1:n
                traj(i) = self.x;
                self.update;
            end
        end
    end
end
