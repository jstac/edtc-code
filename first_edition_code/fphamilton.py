# Filename: fphamilton.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 4.6

from numpy import ones, identity, transpose
from numpy.linalg import solve

pH = ((0.971, 0.029, 0.000),       # Hamilton's kernel
      (0.145, 0.778, 0.077),
      (0.000, 0.508, 0.492))

I = identity(3)                    # 3 by 3 identity matrix
Q, b = ones((3, 3)), ones((3, 1))  # Matrix and vector of ones
A = transpose(I - pH + Q) 
print(solve(A, b))
