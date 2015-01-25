import numpy as np
import scipy as sp
import scipy.sparse as sps
import pdb

def threshold(A):
    A[A < 0] = 0
    A[A > 0] = 1
    return A

def real_sparse_matrix(n, d):
    xs = []
    ys = []
    for c in xrange(n):
        xs += np.random.choice(range(0,n), d, replace=0).tolist()
        ys += [c for i in xrange(d)]
    vals = np.sign(np.random.rand(len(xs)) - 0.5)
    return sps.coo_matrix((vals, (xs,ys)), shape=(n,n)).todense()

def signed_sparse_matrix(n, d):
    G = real_sparse_matrix(n,d)
    return sp.sign(G)

def real_nn(n, l, d):
    G = [real_sparse_matrix(n,d) for i in xrange(l)]
    return np.array(G)

def signed_nn(n, l, d):
    G = [signed_sparse_matrix(n,d) for i in xrange(l)]
    return np.array(G)