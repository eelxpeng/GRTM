import scipy.stats as stats
import numpy as np

def niw(m0, k0, df0, S0, K):
    Sigma = np.array(stats.invwishart.rvs(df0, S0, size=K))
    u = np.zeros((K,m0.shape[0]))
    for i in xrange(K):
        u[i,:] = np.random.multivariate_normal(m0, Sigma[i,:,:] / k0)
    return (u, Sigma)