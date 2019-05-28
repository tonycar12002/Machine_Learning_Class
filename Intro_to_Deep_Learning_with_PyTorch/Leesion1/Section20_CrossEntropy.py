import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Y_minus = []
    P_minus = []
    Y_minus[:] = [1 - x for x in Y]
    P_minus[:] = [1 - x for x in P]
    sigma = Y * np.log(P) + Y_minus * np.log(P_minus)
    return -sum(sigma)