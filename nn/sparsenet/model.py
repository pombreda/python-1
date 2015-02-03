import numpy as np
import scipy as sp
import scipy.sparse as sps
import pdb

from utils import *

'''
def real_sparse_matrix(n, d):
    xs = []
    ys = []
    for c in xrange(n):
        xs += np.random.choice(range(0,n), d, replace=0).tolist()
        ys += [c for i in xrange(d)]
    vals = np.sign(np.random.rand(len(xs)) - 0.5)
    return sps.coo_matrix((vals, (xs,ys)), shape=(n,n)).todense()
'''

def real_sparse_matrix(n, d):
    p = float(d)/float(n)/2.
    A = np.random.random((n,n))
    B = A*(A < p)
    C = A*(A > 1-p)
    G = -np.sign(B) + np.sign(C)
    return G

def signed_sparse_matrix(n, d):
    G = real_sparse_matrix(n,d)
    return sp.sign(G)

def real_nn(n, l, d):
    G = [real_sparse_matrix(n,d) for i in xrange(l)]
    return np.array(G)

def signed_nn(n, l, d):
    G = [signed_sparse_matrix(n,d) for i in xrange(l)]
    return np.array(G)