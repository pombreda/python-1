import numpy as np
import scipy as sp
import pdb

from model import *

def generate_uniform_sparse_hs(n, rho, N):
    hs = []
    while len(hs) < N:
        #rho_curr = min(1, (np.random.random() + 0.7))*rho
        rho_curr = rho
        h = (sp.sign(rho_curr - np.random.random((n,1))) + 1)/2.
        if sum(h):
            hs.append(h)
    return hs

def generate_dirichlet_sparse_hs(n, rho, N):
    hs = []
    alpha = [0.2 for i in xrange(n)]
    hs1 = np.random.dirichlet(alpha, N).T
    #pdb.set_trace()
    for i in xrange(N):
        h = hs1[:,i].reshape((n,1))
        p = np.percentile(h, rho*100)
        h[h > p] = 0
        hs.append(np.sign(h))
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
        h = hs[Ni].copy()
        for li in reversed(range(l)):
            G = Gs[li]
            h = np.sign(np.clip(G.dot(h), 0, 1))
        ys.append(h)

    return ys

def generate_network_and_data(n, l, d, rho, N, which_distribution='uniform'):
    Gs = create_signed_nn(n, l, d)
    if which_distribution == 'uniform':
        hs = generate_uniform_sparse_hs(n, rho, N)
    elif which_distribution == 'dirichlet':
        hs = generate_dirichlet_sparse_hs(n, rho, N)
    else:
        raise 'which_distribution not defined'
    ys = decoder(hs, Gs)
    return ys, hs, Gs

def generate_test_data(Gs, rho, N, which_distribution='uniform'):
    n,m = Gs[0].shape
    assert n == m
    if which_distribution == 'uniform':
        hst = generate_uniform_sparse_hs(n, rho, N)
    elif which_distribution == 'dirichlet':
        hst = generate_dirichlet_sparse_hs(n, rho, N)
    else:
        raise 'which_distribution not defined'
    yst = decoder(hst, Gs)
    return yst, hst