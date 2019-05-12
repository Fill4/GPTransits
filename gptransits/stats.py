import numpy as np
from scipy.stats import mode

def hpd(chain, level):
    
    # Flatten the chain in the walkers and iteration dimensions - x has shape (nwalkers*niter, nparam)
    w = chain.reshape([chain.shape[0]*chain.shape[1], chain.shape[2]])
    # Order the values for each of the paramters
    s = np.sort(w, axis=0)

    # Number of values contained in the desired level interval
    nln = int(np.floor(level*s.shape[0]))
    # Difference between values in all intervals of size nln present in the data
    diff = (s[nln:] - s[:-nln])

    # Get indices where the interval had lowest difference between values
    idx = np.argmin(diff, axis=0)

    # Get values at both sides of the interval chosen
    hpd_down = np.array([s[idx[i],i] for i in range(idx.size)])
    hpd_up = np.array([s[idx[i]+nln,i] for i in range(idx.size)])

    return hpd_down, hpd_up

def mapv(chain, posterior):
    # Get multidimensional index of highest posterior value
    idx = np.unravel_index(np.argmax(posterior), posterior.shape)

    # Get the param array corresponding to the highest posterior value
    map_val = chain[idx[1],idx[0],:]

    return map_val

def mode(chain, bins=50):
    w = chain.reshape([chain.shape[0]*chain.shape[1], chain.shape[2]])

    modes = np.zeros(chain.shape[2])
    for i in range(modes.size):
        counts, edges = np.histogram(w[:,i], bins=bins)
        idx = np.argmax(counts)
        modes[i] = (edges[idx] + edges[idx+1])/2
    
    return modes

