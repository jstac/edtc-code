
import numpy as np
from fvi import LinInterp
from cpdynam import *  # Listing 6.7 in the text
import matplotlib.pyplot as plt

gridsize = 150
grid = np.linspace(a, 35, gridsize)

vals = P(grid)
for i in range(20):
    if i == 0:
        plt.plot(grid, vals, 'k-', label=r'$P$')
    if i == 1:
        plt.plot(grid, vals, 'k--', label=r'$TP$')
    if i == 19:
        plt.plot(grid, vals, 'k-.', label=r'$T^{50}P$')
    p = LinInterp(grid, vals)
    new_vals = [T(p, x) for x in grid]
    vals = new_vals

plt.legend(axespad=0.1)
plt.show()

