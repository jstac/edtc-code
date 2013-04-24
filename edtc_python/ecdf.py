# Filename: ecdf.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 6.3

class ECDF:

    def __init__(self, observations):
        self.observations = observations

    def __call__(self, x):
        counter = 0.0
        for obs in self.observations:
            if obs <= x:
                counter += 1
        return counter / len(self.observations)

