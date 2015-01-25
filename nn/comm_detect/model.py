import numpy as np
import scipy as sp
import pdb

#__all__ = ['create_real_G', 'create_signed_G', 'create_real_nn', 'create_signed_nn']

def threshold(A):
    A[A < 0] = 0
    A[A > 0] = 1
    return A

def create_real_G(n, d):
    p = d/float(n)
    G1 = 2*(np.random.random((n,n)) - 0.5)
    G2 = np.random.random((n,n)) < p
    #pdb.set_trace()
    G = G1*G2
    return G

def create_signed_G(n, d):
    G = create_real_G(n,d)
    return sp.sign(G)

def create_real_nn(n, l, d):
    Gs = []
    for i in xrange(l):
        Gs.append(create_real_G(n,d))
    return Gs

def create_signed_nn(n, l, d):
    Gs = []
    for i in xrange(l):
        Gs.append(create_signed_G(n,d))
    return Gs