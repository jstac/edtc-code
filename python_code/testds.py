# Filename: testds.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 4.3

from ds import DS         # Import from listing 4.2

def quadmap(x):
    return 4 * x * (1 - x)

q = DS(h=quadmap, x=0.1)  # Create an instance q of DS   
T1 = q.trajectory(100)    # T1 holds trajectory from 0.1

q.x = 0.2                 # Reset current state to 0.2
T2 = q.trajectory(100)    # T2 holds trajectory from 0.2

