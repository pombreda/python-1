from numpy.random import (multinomial, \
        dirichlet, gamma, beta)
import numpy as np


def dirichlet_process(alpha, num_samples, G_0=np.random.randn):
    '''
    sample from a dirichlet process
    G_0:    base distribution, we generate k atoms \thetas from G_0
    alpha:  list of \alpha values of length k
    num_samples:   number of samples required
    '''
    k = len(alpha)
    theta = G_0(k)
    p = dirichlet(alpha)
    samples = []
    for i in xrange(num_samples):
        x = multinomial(k, p)
        th = list(set(theta[x]))
        samples.append(th)
    return samples


k = 10
alpha = [7 for i in xrange(k)]
print dirichlet_process(alpha, 10)
