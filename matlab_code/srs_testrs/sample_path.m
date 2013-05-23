% Filename: sample_path.m
% Author: Tomohito Okabe
% Date: March 2013
% Corresponds to: sample_path in Listing 6.1

function path = sample_path(F, phi, X, n)

    global alpha sigma s delta;
    
    n = n-1; % Ajust the last period of path such that  
             % length equals n. 
    
    path(1) = X; 
    
    % Generate path of length n from current state.
    for t= 1:n 
        X_next = update(F, phi(), X);
        path(t+1) = X_next;
        X = X_next;
    end
    
end