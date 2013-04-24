# Filename: cpdynam.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 6.7

from scipy import mean
from scipy.stats import beta
from scipy.optimize import brentq

alpha, a, c = 0.8, 5.0, 2.0                           
W = beta(5, 5).rvs(1000) * c + a   # Shock observations
D = P = lambda x: 1.0 / x   

def fix_point(h, lower, upper):
    """Computes the fixed point of h on [upper, lower]
    using SciPy's brentq routine, which finds the
    zeros (roots) of a univariate function.
    Parameters: h is a function and lower and upper are
    numbers (floats or integers).  """
    return brentq(lambda x: x - h(x), lower, upper)

def T(p, x):
    """Computes Tp(x), where T is the pricing functional
    operator.
    Parameters: p is a vectorized function (i.e., acts 
    pointwise on arrays) and x is a number.  """
    y = alpha * mean(p(W))
    if y <= P(x): 
        return P(x)  
    h = lambda r: alpha * mean(p(alpha*(x - D(r)) + W))
    return fix_point(h, P(x), y)

