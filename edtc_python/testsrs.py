# Filename: testsrs.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 6.2

from srs import SRS               # Import from listing 6.1
from random import lognormvariate 

alpha, sigma, s, delta = 0.5, 0.2, 0.5, 0.1  
# Define F(k, z) = s k^alpha z + (1 - delta) k
F = lambda k, z: s * (k**alpha) * z + (1 - delta) * k 
lognorm = lambda: lognormvariate(0, sigma) 

solow_srs = SRS(F=F, phi=lognorm, X=1.0)
P1 = solow_srs.sample_path(500)   # Generate path from X = 1
solow_srs.X = 60                  # Reset the current state 
P2 = solow_srs.sample_path(500)   # Generate path from X = 60

