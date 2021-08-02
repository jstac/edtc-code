
import numpy as np
from pyx import *

alpha, s, a = 0.5, 0.25, 7
kstar = ((s*a)**(1/(1 - alpha)))

def h(k): 
    return s * a * (k**alpha)

def plotcurve(X, Y, canv):
    for i in range(len(X)-1):
        canv.stroke(path.line(X[i], Y[i], X[i+1], Y[i+1]), [style.linewidth.Thick])

c = canvas.canvas()
upper = 6 
# axes
c.stroke(path.line(0, 0, upper, 0), [deco.earrow(size=0.3)])
c.stroke(path.line(0, 0, 0, upper), [deco.earrow(size=0.3)])
# 45 degrees
c.stroke(path.line(0, 0, upper, upper), [style.linestyle.dashed])
# function curve
X = np.linspace(0, upper, 100)
Y = [h(x) for x in X]
plotcurve(X, Y, c)
# arrows
k = 0.5
for i in range(4):
    c.stroke(path.line(k, k, k, h(k)), 
            [deco.earrow(size=0.15), style.linewidth.Thin])
    c.stroke(path.line(k, h(k), h(k), h(k)), 
            [deco.earrow(size=0.15), style.linewidth.Thin])
    c.stroke(path.line(k, 0, h(k), 0), 
            [deco.earrow(size=0.15), style.linewidth.THIN])
    k = h(k)
k = 5.8
for i in range(3):
    c.stroke(path.line(k, k, k, h(k)), 
            [deco.earrow(size=0.15), style.linewidth.Thin])
    c.stroke(path.line(k, h(k), h(k), h(k)), 
            [deco.earrow(size=0.15), style.linewidth.Thin])
    c.stroke(path.line(k, 0, h(k), 0), 
            [deco.earrow(size=0.15), style.linewidth.THIN])
    k = h(k)

c.stroke(path.line(kstar, 0, kstar, kstar), [style.linestyle.dotted])
c.text(kstar, -0.5, r"k^*", [text.mathmode, text.size.normal])
c.text(upper*1.06, -0.5, r"k_t", [text.mathmode, text.size.normal])
c.text(-0.5, upper*1.06, r"k_{t+1}", [text.mathmode, text.size.normal])
c.writePDFfile("stable45deg")
