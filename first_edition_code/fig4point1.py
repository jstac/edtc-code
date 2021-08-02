
import numpy as np
from pyx import *

xdim = (-10.0, 10.0)
ydim = (-5.0, 5.0)

A1 = np.asarray([[0.58, -0.6],
                   [0.65, 0.3]])

def f(x): return np.dot(A1, x)

# Create a PyX canvas
c = canvas.canvas()

# Set up the axis
c.stroke(path.line(-10, 0, 10, 0), [style.linestyle.dashed])
c.stroke(path.line(0, -5, 0, 5), [style.linestyle.dashed])

# A function to plot arrows
def plotarrow(x):
    fx = f(x)
    c.stroke(path.line(x[0], x[1], fx[0], fx[1]), 
            [style.linewidth.Thick, deco.earrow(size=0.5)])

# A 2 by 2 grid for start points of arrows
xpoints = np.linspace(xdim[0], xdim[1], 14)
ypoints = np.linspace(ydim[0], ydim[1], 8)

for x in xpoints:
    for y in ypoints:
        plotarrow((x, y))

c.writePDFfile("sdsdiagram")
