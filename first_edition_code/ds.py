# Filename: ds.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 4.2

class DS:

    def __init__(self, h=None, x=None):
        """Parameters: h is a function and x is a number
        in S representing the current state."""
        self.h, self.x = h, x

    def update(self):
        "Update the state of the system by applying h."
        self.x = self.h(self.x)

    def trajectory(self, n):
        """Generate a trajectory of length n, starting 
        at the current state."""
        traj = []
        for i in range(n):
            traj.append(self.x)
            self.update()
        return traj
