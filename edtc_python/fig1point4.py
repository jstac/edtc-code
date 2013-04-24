
import numpy as np
import matplotlib.pyplot as plt

theta, alpha = 0.3, 0.4
C = np.log(theta) * alpha
N = 250  
initx = 4
meanx = C / (1 - alpha)

def update(x):
    return C + alpha * x + np.random.randn(1)

path = np.zeros(N)
path[0] = initx
for t in range(N-1):
    path[t+1] = update(path[t])

plt.plot(path, 'k')
plt.axhline(meanx, xmin=0, xmax=N, color='k')
plt.xlabel("time")
plt.ylabel("log income")
plt.show()



