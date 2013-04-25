
import numpy as np
import matplotlib.pyplot as plt

xgrid = np.linspace(0, 1, 100)

h = lambda x, r: r * x * (1 - x)

plt.plot(xgrid, xgrid, '-', color='grey')

r = 0
step = 0.3

while r <= 4:
    y = [h(x, r) for x in xgrid]
    plt.plot(xgrid, y, 'k-')
    r = r + step

plt.show()

