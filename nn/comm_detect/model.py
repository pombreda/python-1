import numpy as np
import scipy as sp
import pdb

def create_real_G(n, d):
    p = d/float(n)
    G1 = 2*(np.random.random((n,n)) - 0.5)
    G2 = (sp.sign(p - np.random.random((n,n))) + 1)/2
    
    G = np.where(G2 > 0, G1, np.zeros((n,n)))
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

def generate_hs(n, rho, N):
    hs = []
    while len(hs) < N:
        h = (sp.sign(rho - np.random.random((n,))) + 1)/2.
        if sum(h):
            hs.append(h)
    return hs

def generate_data(hs, Gs):
    '''
    Gs = l layers of a neural network, arranged as
    G1, G2, ..., Gl
    hs = list of N vectors of length n
    returns ys that are outputs of the network
    '''
    l = len(Gs)
    n, m = Gs[0].shape
    N = len(hs)
    ys = []
    assert n == m
    for Ni in xrange(N):
        h = hs[Ni]
        for li in reversed(range(l)):
            G = Gs[li]
            h = np.sign(np.clip(G.dot(h), 0, 1))
        ys.append(h)

    return ys

def generate_data_auto(n, l, d, rho, N):
    Gs = create_signed_nn(n, l, d)
    hs = generate_hs(n, rho, N)
    ys = generate_data(hs, Gs)
    return ys, hs, Gs
