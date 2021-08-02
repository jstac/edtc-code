---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.9'
    jupytext_version: 1.5.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# The Kolmogorov Forward Equation



## Overview

In this lecture we approach continuous time Markov chains from a more
analytical perspective.

The emphasis will be on describing distribution flows through vector-valued
differential equations.

These distribution flows show how the time $t$ distribution associated with a
given Markov chain $(X_t)$ changes over time.

Density flows will be identified by ordinary differential equations in vector
space that are linear and time homogeneous.

We will see that the solutions of these flows are described by Markov
semigroups.

This leads us back to the theory we have already constructed -- some care will
be taken to clarify all the connections.

In order to avoid being distracted by technicalities, we continue to defer our
treatment of infinite state spaces, assuming throughout this lecture that $|S|
= n$.

As before, $\mathcal D$ is the set of all distributions on $S$.

We will use the following imports

```{code-cell} ipython3
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import quantecon as qe
from numba import njit
from scipy.linalg import expm

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
```

## From Difference Equations to ODEs

{ref}`Previously <invdistflows>` we generated this figure, which shows how distributions evolve over time for the inventory model under a certain parameterization:

```{glue:figure} flow_fig
:name: "flow_fig"

Probability flows for the inventory model.
```

(Hot colors indicate early dates and cool colors denote later dates.)

We also learned how this flow can be described, at least indirectly,
through the Kolmogorov backward equation, which is an ODE.

In this section we examine distribution flows and their connection to 
ODEs and continuous time Markov chains more systematically.

Although our initial discussion appears to be orthogonal to what has come
before, the connections and relationships will soon become clear.


### Review of the Discrete Time Case

Let $(X_t)$ be a discrete time Markov chain with Markov matrix $P$.

{ref}`Recall that <finstatediscretemc>`, in the discrete time case, the distribution $\psi_t$ of $X_t$ updates according to 

$$
    \psi_{t+1} = \psi_t P, 
    \qquad \psi_0 \text{ a given element of } \mathcal D,
$$

where distributions are understood as row vectors.

Here's a visualization for the case $|S|=3$, so that $\mathcal D$ is the unit
simplex in $\mathbb R^3$.

The initial condition is `` (0, 0, 1)`` and the Markov matrix is

```{code-cell} ipython3
P = ((0.9, 0.1, 0.0),
     (0.4, 0.4, 0.2),
     (0.1, 0.1, 0.8))
```

```{code-cell} ipython3
:tags: [hide-input]

def unit_simplex(angle):
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    vtx = [[0, 0, 1],
           [0, 1, 0], 
           [1, 0, 0]]
    
    tri = Poly3DCollection([vtx], color='darkblue', alpha=0.3)
    tri.set_facecolor([0.5, 0.5, 1])
    ax.add_collection3d(tri)

    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), 
           xticks=(1,), yticks=(1,), zticks=(1,))

    ax.set_xticklabels(['$(1, 0, 0)$'], fontsize=12)
    ax.set_yticklabels(['$(0, 1, 0)$'], fontsize=12)
    ax.set_zticklabels(['$(0, 0, 1)$'], fontsize=12)

    ax.xaxis.majorTicks[0].set_pad(15)
    ax.yaxis.majorTicks[0].set_pad(15)
    ax.zaxis.majorTicks[0].set_pad(35)

    ax.view_init(30, angle)

    # Move axis to origin
    ax.xaxis._axinfo['juggled'] = (0, 0, 0)
    ax.yaxis._axinfo['juggled'] = (1, 1, 1)
    ax.zaxis._axinfo['juggled'] = (2, 2, 0)
    
    ax.grid(False)
    
    return ax


def convergence_plot(ψ, n=14, angle=50):

    ax = unit_simplex(angle)

    P = ((0.9, 0.1, 0.0),
         (0.4, 0.4, 0.2),
         (0.1, 0.1, 0.8))
    
    P = np.array(P)

    x_vals, y_vals, z_vals = [], [], []
    for t in range(n):
        x_vals.append(ψ[0])
        y_vals.append(ψ[1])
        z_vals.append(ψ[2])
        ψ = ψ @ P

    ax.scatter(x_vals, y_vals, z_vals, c='darkred', s=80, alpha=0.7, depthshade=False)

    return ψ

ψ = convergence_plot((0, 0, 1))

plt.show()
```

There's a sense in which a discrete time Markov chain "is" a homogeneous
linear difference equation in distribution space.

To clarify this, suppose we 
take $G$ to be a linear map from $\mathcal D$ to itself and
write down the difference equation 

$$
    \psi_{t+1} = G(\psi_t)
    \quad \text{with } \psi_0 \in \mathcal D \text{ given}.
$$ (gdiff)

Because $G$ is a linear map from a finite dimensional space to itself, it can
be represented by a matrix.

Moreover, a matrix $P$ is a Markov matrix if and only if $\psi \mapsto
\psi P$ sends $\mathcal D$ into itself (check it if you haven't already).

So, under the stated conditions, our difference equation {eq}`gdiff` uniquely
identifies a Markov matrix, along with an initial condition $\psi_0$.

Together, these objects identify the joint distribution of a discrete time Markov chain, as {ref}`previously described <jdfin>`.


### Shifting to Continuous Time

We have just argued that a discrete time Markov chain "is" a linear difference
equation evolving in $\mathcal D$.

This strongly suggests that a continuous time Markov chain "is" a linear ODE
evolving in $\mathcal D$.

In this scenario,

1. distributions update according to an automous linear differential equation and
2. the vector field is such that trajectories remain in $\mathcal D$.

This intuition is correct and highly beneficial.  

The rest of the lecture maps out the main ideas.



## ODEs in Distribution Space

Consider linear differential equation given by 

$$
    \psi_t' = \psi_t Q, 
    \qquad \psi_0 \text{ a given element of } \mathcal D,
$$ (ode_mc)

where 

* $Q$ is an $n \times n$ matrix with suitable properties, 
* distributions are again understood as row vectors, and
* derivatives are taken element by element, so that

$$
    \psi_t' =
    \begin{pmatrix}
        \frac{d}{dt} \psi_t(1) &
        \cdots &
        \frac{d}{dt} \psi_t(n)
    \end{pmatrix}
$$


Using the matrix exponential, the unique solution to the initial value problem
{eq}`ode_mc` can be expressed as

$$
    \psi_t = \psi_0 P_t 
    \quad \text{where } P_t := e^{tQ}
$$ (cmc_sol)

To check this, we use {eq}`expoderiv` again to get

$$
    \frac{d}{d t} P_t =  Q e^{tQ} = e^{tQ} Q 
$$

Recall that the first equality can be written as 
$\frac{d}{d t} P_t =  Q P_t$ and this is the Kolmogorov backward equation.  

The second equality can be written as 

$$
    \frac{d}{d t} P_t = P_t Q 
$$

and is called the **Kolmogorov forward equation**.

With $\psi_t$ set to $\psi_0 P_t$ and applying the Kolmogorov forward
equation, we obtain

$$
    \frac{d}{d t} \psi_t 
    = \psi_0 \frac{d}{d t} P_t 
    = \psi_0 P_t Q
    = \psi_t Q
$$

This confirms that {eq}`cmc_sol` solves {eq}`ode_mc`.


Here's an example of a distribution flow created by {eq}`ode_mc` with 
initial condition `` (0, 0, 1)`` and

```{code-cell} ipython3
Q = ((2, -3, 1),
     (3, -5, 2),
     (1, -4, 3))
```

```{code-cell} ipython3
:tags: [hide-input]

Q = np.array(Q)

def P_t(ψ, t):
    return ψ @ expm(t * Q)

def flow_plot(ψ, h=0.01, n=200, angle=50):

    ax = unit_simplex(angle)

    Q = ((2, -3, 1),
         (3, -5, 2),
         (1, -4, 3))
    Q = np.array(Q)
    
    x_vals, y_vals, z_vals = [], [], []
    for t in range(n):
        x_vals.append(ψ[0])
        y_vals.append(ψ[1])
        z_vals.append(ψ[2])
        ψ = P_t(ψ, h)

    ax.scatter(x_vals, y_vals, z_vals, c='darkred', s=80, alpha=0.7, depthshade=False)

    return ψ

ψ = flow_plot(np.array((0, 0, 1)))

plt.show()
```

(We use Euler discretization to trace out the flow.)


In this calculation, $Q$ was chosen with some care, so that the flow remains
in $\mathcal D$.

This raises a key question: what properties do we require on $Q$ such that
$\psi_t$ is always in $\mathcal D$?

We seek necessary and sufficient conditions, so we can determine
exactly the set of continuous time Markov models on our state space.

We will answer this question in stages.



#### Preserving Distributions

Recall that the linear update rule $\psi \mapsto \psi P$ is invariant on
$\mathcal D$
if and only if $P$ is a Markov matrix.

So now we can rephrase our key question regarding invariance on $\mathcal D$:

What properties do we need to impose on $Q$ so that $P_t$ is a Markov matrix
for all $t$?

A square matrix $Q$ is called a **transition rate matrix** if $Q$ has zero row
sums and $Q(i, j) \geq 0$ whenever $i \not= j$.

(Some authors call a transition rate matrix a $Q$ matrix.)

Having zero row sums can be expressed as $Q \mathbb 1 = 0$.

As a small exercise, you can check that the following is true

$$
    Q \text{ has zero row sums }
    \iff
    Q^k \mathbb 1 = 0 \text{ for all } k \geq 1
$$ (zrsnec)

**Theorem** If $Q$ is an $n \times n$ matrix and $P_t := e^{tQ}$, then the
following statements are equivalent:

1. $P_t$ is a Markov matrix for all $t$.
1. $Q$ is a transition rate matrix.

*Proof:*  Suppose first that $Q$ is a transition rate matrix and set $P_t =
e^{tQ}$ for all $t$.

By the definition of the exponential function, for all $t \geq 0$,

$$
    P_t \mathbb 1 = \mathbb 1 + tQ \mathbb 1 + \frac{1}{2!} t^2 Q^2 \mathbb 1 + \cdots
$$

From {eq}`zrsnec`, we see that $P_t$ has unit row sums.

As a second observation, note that, for any $i, j$ and $t \geq 0$,

$$
    P_t(i, j) = \mathbb 1\{i = j\} + t Q(i, j) + o(t)
$$ (otp)

From {eq}`otp`, both off-diagonal and on-diagonal elements of $P_t$ are nonnegative.

Hence $P_t$ is a Markov matrix.

Regarding the converse implication, suppose that $P_t = e^{tQ}$ is a Markov
matrix for all $t$.

Because $P_t$ has unit row sums and differentiation is linear, 
we can employ the Kolmogorov backward equation to obtain

$$
    Q  \mathbb 1
      = Q P_t \mathbb 1
      = \left( \frac{d}{d t} P_t \right) \mathbb 1
      = \frac{d}{d t} (P_t \mathbb 1)
      = \frac{d}{d t} \mathbb 1
      = 0
$$

Hence $Q$ has zero row sums.

Moreover, in view of {eq}`otp`, the off diagonal elements of $Q$ must be positive.

Hence $Q$ is a transition rate matrix.

This completes the proof.








### Representations

Both the semigroup and its infintessimal generator are natural representations
of a given continuous time Marko chain.

The semigroup can be constructed uniquely constructed from its generator $Q$
via $P_t = e^{tQ}$, and the generator can be recovered from the semigroup via

$$
    Q = \frac{d}{d t} P_t \big|_0
$$

The last claim follows from $P_t = I$ and either the forward or backward
Kolmogorov equation.

The semigroup is, in some sense, more informative than the generator, since it
allows us to update distributions to any point in time.

But the generator is simpler and often more intuitive in particular
applications.

### The Inventory Example

Let's go back to the inventory example we discussed above.

What is the infintessimal generator for this problem?

The intuitive interpretation of a given generator $Q$ is that

$$
    Q(i, j) = \text{ rate of flow from state $i$ to state $j$}
$$

For example, if we observe the inventories of many firms independently
following the model above, then $Q(i, j) = r$ means that firms inventories
transition from state $i$ to state $j$ at a rate of $r$ per unit of time.

[improve this, make it clearer]

## Jump Chains: The General Case

In this section we provide a natural way to construct continuous time
continuous time Markov chain on our finite state space $S$.

Later we will show that *every* continuous time Markov chain on a finite
state space can be represented in this way.

Intro model based on $\lambda$ fixed and $\Pi(x, y)$.

Build $Q$ from this model, and then $P_t$ from $Q$ in the usual way.

Build $P_t$ directly using probabilistic reasoning and show that the two
coincide.




## All CTMCs are Jump Chains 

Start with a Q matrix and construct the jump chain.

When does $Q$ admit the decomposition $Q = \lambda (\Pi - I)$?





## The Gillespie Algorithm

Exponential clocks.






## Exercises

The Markov semigroup properties lead to the KF and KB equations.

When treating Kolmogorov forward and backward equations, do a first order
Euler discretization and link to the discrete case.



## Solutions

To be added.

```{code-cell} ipython3

```
