# Filename: ar1.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 8.1

from random import normalvariate as N

a, b = 0.5, 1       # Parameters
X = {}              # An empty dictionary to store path
X[0] = N(0, 1)      # X_0 has distribution N(0, 1)

for t in range(100):
    X[t+1] = N(a * X[t] + b, 1)


