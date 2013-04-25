% Filename: edf.m
% Author: Andy Qi
% Date: December 2008
% Corresponds to: Listing xxxx
% Corresponds to Listing 6.3 in Economic Dynamics
% Author: Andy Qi, Kyoto University
% Date: Dec 22,2008

classdef edf
    properties
        observations;
    end
    
    methods
        function obj = edf(observations)
            obj.observations = observations;
        end
        
        function f = p(obj,x)
            counter = 0;
            for obs = obj.observations
                if obs <= x
                    counter = counter + 1;
                end
            end
            f = counter / length(obj.observations);
        end
    end
end
