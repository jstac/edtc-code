# Filename: quadmap1.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 4.1

from pylab import plot, show  # Requires Matplotlib

datapoints = []               # Stores trajectory
x = 0.11                      # Initial condition
for t in range(200):
    datapoints.append(x)
    x = 4 * x * (1 - x)

plot(datapoints)
show()
