# Filename: polyclass0.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing xxxx

class Polynomial:

    def initialize(coef):
        """Creates an instance p of the Polynomial class,
        where p(x) = coef[0] x^0 + ... + coef[N] x^N."""

    def evaluate(x):
        y = sum(a*x**i for i, a in enumerate(coef))
        return y

    def differentiate():
        new_coef = [i*a for i, a in enumerate(coef)]
        # Remove the first element, which is zero
        del new_coef[0]  
        # And reset coefficients data to new values
        coef = new_coef

