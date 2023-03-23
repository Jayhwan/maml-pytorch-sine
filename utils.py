
import numpy as np

def simplex_proj(beta):
    beta_sorted = np.flip(np.sort(beta))
    rho = 1
    for i in range(len(beta)-1):
        j = len(beta) - i
        test = beta_sorted[j-1] + (1 - np.sum(beta_sorted[:j]))/(j)
        if test > 0:
            rho = j
            break

    lam = (1-np.sum(beta_sorted[:rho]))/(rho)
    return np.maximum(beta + lam,0)
