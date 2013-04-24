
import matplotlib.pyplot as plt
from math import exp, sqrt
from random import normalvariate

alpha = 0.5
s = 0.25
a1 = 15 
a2 = 25
ss = sqrt(0.02)
kb = 24.1

def update(k):
    a = a1 if k < kb else a2
    return s * a * (k**alpha) * exp(normalvariate(0, ss))

def ts(init, T): 
    k = init
    path = [k]
    for t in range(T):
        k = update(k)
        path.append(k)
    return path

L = 500
plt.plot(ts(1, L), 'k-')
plt.plot(ts(60, L), '-', color='grey')
plt.xlabel('time')
plt.ylabel('investment')
plt.show()
