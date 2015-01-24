import numpy as np
import scipy as sp
import pdb

from model import *

def generate_uniform_sparse_hs(n, rho, N):
    hs = []
    while len(hs) < N:
        rho_curr = min(1, (np.random.random() + 0.7))*rho
        h = (sp.sign(rho_curr - np.random.random((n,1))) + 1)/2.
        if sum(h):
            hs.append(h)
    return hs

def decoder(hs, Gs):
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

def generate_network_and_data(n, l, d, rho, N):
    Gs = create_signed_nn(n, l, d)
    hs = generate_uniform_sparse_hs(n, rho, N)
    ys = decoder(hs, Gs)
    return ys, hs, Gs

def generate_test_data(Gs, rho, N):
    n,m = Gs[0].shape
    assert n == m
    hst = generate_uniform_sparse_hs(n, rho, N)
    yst = decoder(hst, Gs)
    return yst, hst