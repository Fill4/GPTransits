import numpy as np

# -------------------- CONVERGENCE METRICS -------------------------
def gelman_rubin(chain):
    """
    Computes the Gelman-Rubin PSRF for each of the parameters sampled, assuming all parameters are independent.

    Parameters
    ----------
        chain : array-like 
            Array of [nwalkers, niterations, nparameters] output from the sampling algorithm

    Returns
    -------
        R_hat : array-like
            Array of [nparameters] with the R_hat diagnostic for each parameter
    """
    m, n, p = chain.shape

    # Between chain variance
    B_over_n = np.sum((np.mean(chain,1) - np.mean(chain, (0,1)))**2, 0) / (m-1)

    # Within chain variance
    W = np.sum((chain - np.mean(chain,1,keepdims=True))**2, (0,1)) / (m*(n-1))

    # Unbiased estimate of the true variance
    var_plus = W*(n-1)/n + B_over_n
    
    # Pooled posterior variance estimate
    V = var_plus + B_over_n/m

    # Potencial scale reduction factor (PSRF). R = np.sqrt(V/W) ??
    R_hat = V/W, V, W

    return R_hat


def gelman_brooks(chain):
    """
    Computes the Gelman-Brooks multivariate PSRF that tests for convergence when there is dependency between parameters.

    Parameters
    ----------
        chain : array-like 
            Array of [nwalkers, niterations, nparameters] output from the sampling algorithm

    Returns
    -------
        R_hat : array-like
            Array of [nparameters] where each value representes the mPSRF diagnostic computed for each eigenvalue of the covariance matrices
    """
    m, n, p = chain.shape

    # Between chain covariance matrix
    b2 = (np.mean(chain,1) - np.mean(chain, (0,1)))
    B_over_n = np.dot(b2.T, b2) / (m-1)

    # Within chain covariance matrix
    w2 = (chain - np.mean(chain,1,keepdims=True))
    W = np.tensordot(w2, w2, axes=((0,1),(0,1))) / (m*(n-1))

    # Pooled posterior variance-covariance matrix
    V = W*((n-1)/n) + B_over_n*(1+(1/m))

    # Find eigenvalues of the matrix (W^-1*B/n) and pick the largest one
    W_inv = np.linalg.inv(W)
    eigenvalues = np.linalg.eigvals(W_inv * B_over_n)

    # Compute the multivariate PSRF using the eigenvalue found
    R_hat = ((n-1)/n) + (1+(1/m)) * eigenvalues

    return R_hat


def geweke(chain, frac1=0.1, frac2=0.5, bins=10):
    """
    Computes the Geweke z-score for each walker in the chain

    Parameters
    ----------
        chain : array-like 
            Array of [nwalkers, niterations, nparameters] output from the sampling algorithm

    Returns
    -------
        zscores : array-like
            Array of [nwalkers, nparameters] where each value is the Geweke z-score for the interval of each walker
    """
    zscores = np.full([bins, chain.shape[0], chain.shape[2]], 100.0)

    starts = np.linspace(0, chain.shape[1]/2, num=bins).astype(int)

    for i in range(starts.size):
        r_chain = chain[:,starts[i]:,:]
        means = np.mean(r_chain[:,:int(frac1*r_chain.shape[1]),:], axis=1) - np.mean(r_chain[:,int(frac2*r_chain.shape[1]):,:], axis=1)
        variances = np.sqrt(np.var(r_chain[:,:int(frac1*r_chain.shape[1]),:], axis=1) + np.var(r_chain[:,int(frac2*r_chain.shape[1]):,:], axis=1))
        zscores[i] = abs(means / variances)

    return starts, zscores




