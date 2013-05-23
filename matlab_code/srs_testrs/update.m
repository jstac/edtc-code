% Filename: update.m
% Author: Tomohito Okabe
% Date: March 2013
% Corresponds to: update in Listing 6.1

function X_next = update(F, phi, X)

    global alpha sigma s delta ;

    X_next = F(X,phi);

end
