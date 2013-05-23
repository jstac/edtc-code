# Filename: genfinitemc.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 4.4

from random import uniform

def sample(phi):
    """Returns i with probability phi[i], where phi is an
    array (e.g., list or tuple)."""
    a = 0.0
    U = uniform(0,1)  
    for i in range(len(phi)):
        if a < U <= a + phi[i]:
            return i
        a = a + phi[i]


class MC:
    """For generating sample paths of finite Markov chains 
    on state space S = {0,...,N-1}."""
    
    def __init__(self, p=None, X=None):
        """Create an instance with stochastic kernel p and 
        current state X. Here p[x] is an array of length N
        for each x, and represents p(x,dy).  
        The parameter X is an integer in S."""
        self.p, self.X = p, X

    def update(self):
        "Update the state by drawing from p(X,dy)."
        self.X = sample(self.p[self.X])  

    def sample_path(self, n):
        """Generate a sample path of length n, starting from 
        the current state."""
        path = []
        for i in range(n):
            path.append(self.X)
            self.update()
        return path


