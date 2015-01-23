import numpy as np
import scipy as sp
import pdb

from model import *
from generate_data import *


def create_correlation_matrix(rho, ys, eps):
    N = len(ys)
    n = len(ys[0])

    Gamma = np.zeros((n,n))
    for i in xrange(N):
        mat = ys[i].dot(ys[i].T)

        # threshold low correlations
        #mat[np.abs(mat) < eps] = 0    
        Gamma += mat
    
    correlations = np.ones((n,n))
    correlations[Gamma < rho*N/3.] = 0
    return correlations

def find_positive_edges(correlations):
    n, _ = correlations.shape

