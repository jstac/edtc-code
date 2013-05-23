# Filename: p2srs.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 5.3

def createF(p):
    """Takes a kernel p on S = {0,...,N-1} and returns a 
    function F(x,z) which represents it as an SRS. 
    Parameters: p is a sequence of sequences, so that p[x][y] 
    represents p(x,y) for x,y in S.
    Returns: A function F with arguments (x,z)."""
    S = range(len(p[0]))
    def F(x,z):
        a = 0
        for y in S:
            if a < z <= a + p[x][y]: 
                return y
            a = a + p[x][y]
    return F


