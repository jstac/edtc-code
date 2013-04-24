# Filename: srs.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 6.1

class SRS:
    
    def __init__(self, F=None, phi=None, X=None):
        """Represents X_{t+1} = F(X_t, W_{t+1}); W ~ phi.
        Parameters: F and phi are functions, where phi() 
        returns a draw from phi. X is a number representing 
        the initial condition."""
        self.F, self.phi, self.X = F, phi, X

    def update(self):
        "Update the state according to X = F(X, W)."
        self.X = self.F(self.X, self.phi())

    def sample_path(self, n):
        "Generate path of length n from current state."
        path = []
        for i in range(n):
            path.append(self.X)
            self.update()
        return path
