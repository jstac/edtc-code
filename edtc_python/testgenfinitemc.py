# Filename: testgenfinitemc.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 4.5

from genfinitemc import sample, MC  # Import from listing 4.4

pH = ((0.971, 0.029, 0.000),  
      (0.145, 0.778, 0.077), 
      (0.000, 0.508, 0.492))

psi = (0.3, 0.4, 0.3)        # Initial condition
h = MC(p=pH, X=sample(psi))  # Create an instance of class MC
T1 = h.sample_path(1000)     # Series is Markov-(p, psi)

psi2 = (0.8, 0.1, 0.1)       # Alternative initial condition
h.X = sample(psi2)           # Reset the current state
T2 = h.sample_path(1000)     # Series is Markov-(p, psi2)

