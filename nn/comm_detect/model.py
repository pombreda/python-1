import numpy as np
import scipy as sp
import pdb

def threshold(A):
    A[A < 0] = 0
    A[A > 0] = 1
    return A

def real_sparse_matrix(n, d):
    p = d/float(n)
    G1 = 2*(np.random.random((n,n)) - 0.5)
    G2 = np.random.random((n,n)) < p
    #pdb.set_trace()
    g = G1*G2
    return g

def signed_sparse_matrix(n, d):
    G = real_sparse_matrix(n,d)
    return sp.sign(G)

def real_nn(n, l, d):
    G = [real_sparse_matrix(n,d) for i in xrange(l)]
    return np.array(G)

def signed_nn(n, l, d):
    G = [signed_sparse_matrix(n,d) for i in xrange(l)]
    return np.array(G)