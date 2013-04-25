# Filename: fvi.py
# Author: John Stachurski
# Date: August 2009
# Corresponds to: Listing 6.5

from scipy import linspace, mean, exp, randn 
from scipy.optimize import fminbound
from lininterp import LinInterp        # From listing 6.4

theta, alpha, rho = 0.5, 0.8, 0.9      # Parameters
def U(c): return 1 - exp(- theta * c)  # Utility
def f(k, z): return (k**alpha) * z     # Production 
W = exp(randn(1000))                   # Draws of shock

gridmax, gridsize = 8, 150
grid = linspace(0, gridmax**1e-1, gridsize)**10

def maximum(h, a, b):
    return float(h(fminbound(lambda x: -h(x), a, b)))

def bellman(w):
    """The approximate Bellman operator.
    Parameters: w is a vectorized function (i.e., a 
    callable object which acts pointwise on arrays).
    Returns: An instance of LinInterp.
    """
    vals = []
    for y in grid:
        h = lambda k: U(y - k) + rho * mean(w(f(k,W)))
        vals.append(maximum(h, 0, y))
    return LinInterp(grid, vals)


