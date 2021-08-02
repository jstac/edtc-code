from ds import DS
from matplotlib.pyplot import plot, show, xlabel

q = DS(h=None, x=0.1)

r = 2.5
while r < 4:
    q.h = lambda x: r * x * (1 - x)
    t = q.trajectory(1000)[950:]
    plot([r] * len(t), t, 'k.', ms=0.4)
    r = r + 0.005

xlabel(r'$r$', fontsize=16)
show()


