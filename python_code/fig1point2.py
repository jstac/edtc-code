from pyx import *
from pyx.deco import earrow
text.set(mode="latex")
text.preamble(r"\usepackage{times}")
from scipy import log, sqrt, pi, linspace, exp

theta = 0.3
alpha = 0.4
c = (log(theta)) * alpha
muinit = -2 
varinit = 0.8
numden = 7

updatemu = lambda m: c + alpha * m
updatevar = lambda v:  alpha**2 * v + 1

def phi(z, mu, var): 
    return exp(- (z - mu)**2 / (2.0 * var)) / sqrt(2 * pi * var)

gridmin, gridmax = -3.2, 1 
g = graph.graphxy(width=10, x=graph.axis.lin(min=gridmin, max=gridmax))

x_grid = linspace(gridmin, gridmax, 100)
mu, var = muinit, varinit
i = 1
gr = 0.7
step = gr / numden
while i <= numden:
    plotpairs = [(x, phi(x, mu, var)) for x in x_grid]
    g.plot(graph.data.points(plotpairs, 
            x=1, y=2),
            [graph.style.line([color.gray(gr)])])
    mu, var = updatemu(mu), updatevar(var)
    gr -= step
    i += 1

x1, y1 = g.pos(-1.3, 0.1)
x2, y2 = g.pos(-0.8, 0.17)
g.stroke(path.line(x1, y1, x2, y2), [earrow.normal])
g.text(x1-0.1, y1-0.3, r"$x_t \sim N(-2, 0.8)$", [text.halign.right])

g.writePDFfile("normaldensities")

