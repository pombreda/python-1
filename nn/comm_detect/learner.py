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
    
    C = np.ones((n,n))
    C[Gamma < rho*N/3.] = 0
    return C

def find_positive_edges(d, C):
    n, _ = C.shape
    G = np.zeros((n,n))
    hcounter = 0
    #pdb.set_trace()
    for u in xrange(n):
        if C[u,u] > 0:
            # FIX: where [0] because where returns a tuple
            Su = set(np.where(C[u,:] > 0)[0].tolist())    # all siblings of u, say v
            for v in Su:
                if not v == u:
                    Sv = set(np.where(C[v,:] > 0)[0].tolist())
                    S = list(Su.intersection(Sv))
                    if len(S) < 1.3*d:
                        #print 'found common cause'
                        # create a parent to explain the cause
                        #print 'S:', list(S)
                        G[S,hcounter] = 1
                        hcounter += 1
                        # delete all correlations in S
                        C[u,S] = 0
                        break
    return G
    