
from fpi import *
import matplotlib.pyplot as plt
import numpy as np

tol = 0.005
sigma = get_greedy(U)
v = get_value(sigma, U)

while 1:
    plt.plot(grid, sigma(grid))
    sigma_new = get_greedy(v)
    v_new = get_value(sigma_new, v)
    if np.max(abs(v(grid) - v_new(grid))) < tol:
        break
    else:
        sigma = sigma_new
        v = v_new

plt.show()
