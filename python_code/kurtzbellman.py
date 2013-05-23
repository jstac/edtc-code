# Filename: kurtzbellman.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 5.1

beta, rho, B, M = 0.5, 0.9, 10, 5
S = range(B + M + 1)  # State space = 0,...,B + M
Z = range(B + 1)      # Shock space = 0,...,B 

def U(c):      
    "Utility function."
    return c**beta

def phi(z):    
    "Probability mass function, uniform distribution."
    return 1.0 / len(Z) if 0 <= z <= B else 0

def Gamma(x):  
    "The correspondence of feasible actions."
    return range(min(x, M) + 1)

def T(v):      
    """An implementation of the Bellman operator.
    Parameters: v is a sequence representing a function on S.
    Returns: Tv, a list."""
    Tv = []        
    for x in S:
        # Compute the value of the objective function for each 
        # a in Gamma(x), and store the result in vals
        vals = []   
        for a in Gamma(x):
            y = U(x - a) + rho * sum(v[a + z]*phi(z) for z in Z)
            vals.append(y)
        # Store the maximum reward for this x in the list Tv
        Tv.append(max(vals))
    return Tv 

