# Filename: kurtzvsigma.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 5.2

from numpy import zeros, dot, array
from kurtzbellman import S, rho, phi, U  # From listing 5.1

def value_of_policy(sigma):
    "Computes the value of following policy sigma."

    # Set up the stochastic kernel p_sigma as a 2D array:
    N = len(S)
    p_sigma = zeros((N, N))   
    for x in S:
        for y in S: 
            p_sigma[x, y] = phi(y - sigma[x])

    # Create the right Markov operator M_sigma:
    M_sigma = lambda h: dot(p_sigma, h)

    # Set up the function r_sigma as an array:
    r_sigma = array([U(x - sigma[x]) for x in S])
    # Reshape r_sigma into a column vector:
    r_sigma = r_sigma.reshape((N, 1))

    # Initialize v_sigma to zero:
    v_sigma = zeros((N,1))
    # Initialize the discount factor to 1:
    discount = 1

    for i in range(50):
        v_sigma = v_sigma + discount * r_sigma 
        r_sigma = M_sigma(r_sigma)
        discount = discount * rho

    return v_sigma

