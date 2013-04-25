# Filename: fpi.py
# Author: John Stachurski
# Date: August 2009
# Corresponds to: Listing 6.6

from fvi import *  # Import all definitions from listing 6.5
from scipy import absolute as abs

def maximizer(h, a, b):
    return float(fminbound(lambda x: -h(x), a, b))

def T(sigma, w):
    "Implements the operator L T_sigma."
    vals = []
    for y in grid:
        Tw_y = U(y - sigma(y)) + rho * mean(w(f(sigma(y), W)))
        vals.append(Tw_y)
    return LinInterp(grid, vals)

def get_greedy(w):
    "Computes a w-greedy policy."
    vals = []
    for y in grid:
        h = lambda k: U(y - k) + rho * mean(w(f(k, W)))
        vals.append(maximizer(h, 0, y))
    return LinInterp(grid, vals)

def get_value(sigma, v):    
    """Computes an approximation to v_sigma, the value
    of following policy sigma. Function v is a guess.
    """
    tol = 1e-2         # Error tolerance 
    while 1:
        new_v = T(sigma, v)
        err = max(abs(new_v(grid) - v(grid)))
        if err < tol:
            return new_v            
        v = new_v
