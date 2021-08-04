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

# Chapter 2 Code

+++

We'll use Python's NumPy library in what follows, as well as Matplotlib for plotting.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

### Bisection

+++

Here's an implementation of the bisection algorithm.  We will test it on this function:

```{code-cell} ipython3
f = lambda x: np.sin(4 * (x - 1/4)) + x + x**20 - 1
```

The implementation:

```{code-cell} ipython3
M = 1000
ϵ = 1e-8
α, β = 0, 1

    
i = 1
a, b = α, β
while i <= M:
    c = (a + b) / 2
    if abs(f(c)) < ϵ:
        print(c)
        break
    i += 1
    if f(a) * f(c) < 0:
        b = c
    else:
        a = c
        
if i > M:
    print("Failed to converge.")
```

### User Defined Functions

+++

Here's the basic implementation of the function $\tau$.

```{code-cell} ipython3
def τ(z, S, ϕ):
    """
    Evaluates the function τ(z) given data S, ϕ, where
    S and ϕ are assumed to be arrays.
    """
    a = 0
    for i, x in enumerate(S):
        b = a + ϕ[i]
        if a < z <= b:
            return x
        a = b
```

Here's a more efficient implementation.

```{code-cell} ipython3
def τ(z, S, ϕ):
    i = np.searchsorted(np.cumsum(ϕ), z)
    return S[i]
```

And here's a closure that generates the function $\tau$.

```{code-cell} ipython3
def τ_factory(S, ϕ):
    Φ = np.cumsum(ϕ)
    
    def τ(z):
        i = np.searchsorted(Φ, z)
        return S[i]
    
    return τ
```

We generate a function $\tau$ that acts on $z$ alone by calling the function factory:

```{code-cell} ipython3
ϕ = 0.2, 0.5, 0.3
S = 0, 1, 2
τ = τ_factory(S, ϕ)
```

```{code-cell} ipython3
τ(0.1)  # Should be 0
```

All of these functions work as expected.  To illustrate, here $\tau$ is used to generate draws from a given distribution $\phi$.  

```{code-cell} ipython3

size = 100_000

draws = np.empty(size)
for j in range(size):
    W = np.random.uniform()
    draws[j] = τ(W)
    
# Compute fraction of draws with each possible value
frequency = [np.mean(draws==j) for j in S]
    
```

Let's check that the empirical frequency approximately coincides with the probabilities in $\phi$.

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.bar(S, frequency, 
       edgecolor='k',
       facecolor='b',
       alpha=0.25, 
       label="histogram")

ax.stem(S, ϕ, label='$\\phi$')

ax.legend()

plt.show()
```

### Object Oriented Programming

+++

Here's a class that implements the function $\tau$ as a method, as well as a method to generate draws from $\phi$.

```{code-cell} ipython3
class Tau:
    
    def __init__(self, S, ϕ):
        self.S = S
        self.Φ = np.cumsum(ϕ)
    
    def τ(self, z):
        i = np.searchsorted(self.Φ, z)
        return self.S[i]
    
    def draw(self):
        W = np.random.uniform()
        return self.τ(W)
```

```{code-cell} ipython3
τ = Tau(S, ϕ)
```

```{code-cell} ipython3
for i in range(5):
    print(τ.draw())
```

```{code-cell} ipython3

```
