---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Chapter 4 Code

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

xmin, xmax = -10.0, 10.0
ymin, ymax = -5.0,  5.0


A1 = np.asarray([[0.55, -0.6],
                 [0.5, 0.4]])

def f(x, y): 
    return  A1 @ (x, y)

def draw_arrow(x, y, ax):
    eps = 1.0
    v1, v2 = f(x, y)
    nrm = np.sqrt(v1**2 + v2**2)
    scale = eps / nrm
    ax.arrow(x, y, scale * v1, scale * v2,
            antialiased=True, 
            alpha=0.4,
            head_length=0.025*(xmax - xmin), 
            head_width=0.012*(xmax - xmin),
            fill=False)

xgrid = np.linspace(xmin * 1.1, xmax * 0.95, 20)
ygrid = np.linspace(ymin * 1.1, ymax * 0.95, 20)

fig, ax = plt.subplots()

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_xticks((0,))
ax.set_yticks((0,))
ax.grid()

for x in xgrid:
    for y in ygrid:
        draw_arrow(x, y, ax)

plt.show()
```

```{code-cell} ipython3

```
