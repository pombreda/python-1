import numpy as np
import scipy as sp
import pdb

from model import *
from generate_data import *


def create_correlation_matrix(rho, Y, eps):
    N, n = Y.shape

    C = np.zeros((n,n))
    for i in xrange(N):
        yt = np.atleast_2d(Y[i,:])
        C += (yt.T).dot(y)

    C = C*(C > rho*N/3.)
    return C

def find_positive_edges(d, C):
    n, _ = C.shape
    G = np.zeros((n,n))
    hcounter = 0
    for u in xrange(n):
        if C[u,u] > 0:
            # FIX: where [0] because where returns a tuple
            Su = set(np.where(C[u,:] > 0)[0].tolist())    # all siblings of u, say v
            for v in Su:
                if not v == u:
                    Sv = set(np.where(C[v,:] > 0)[0].tolist())
                    S = list(Su.intersection(Sv))
                    if len(S) < 1.3*d + 1:
                        #print 'found common cause'
                        # create a parent to explain the cause
                        #print 'S:', list(S)
                        G[S,hcounter] = 1
                        hcounter += 1
                        # delete all correlations in S
                        C[u,S] = 0
                        break
    return G
    

def find_negative_edges(hsp, Y, G):
    N, n = Y.shape
    R = -1*np.ones((n,n))
    R += G
    #pdb.set_trace()
    backedges = [set((np.nonzero(G[i,:])[0]).tolist()) for i in xrange(n)]
    
    for h, y in zip(hsp, Y):
        ones_in_y, _ = np.nonzero(y)
        supph = set(np.nonzero(h)[0].tolist())
        for u in ones_in_y:
            if len(supph.intersection(backedges[u])) == 1:
                R[u,list(supph)] = 0
    return R

def learn_network(n, l, d, rho, ys):
    ysc = ys
    hs = None
    Gs = []
    for i in xrange(l):
        rhoi = rho*(d/2.)**(l- (i+1))
        C = create_correlation_matrix(rho=rhoi, ys=ysc, eps=1e-3)
        G = find_positive_edges(d=d, C=C)
        hsp = encode_one(d, G, ysc)
        R = find_negative_edges(hsp, ysc, G)
        #pdb.set_trace()
        G += R
        Gs.append(G)

        ysc = hsp
        hs = hsp
    return np.array(Gs), hs

def encode_one(d, G, ys):
    N = len(ys)
    #pdb.set_trace()
    n,m = ys[0].shape
    hs = []
    Gt = G.T
    for i in xrange(N):
        h = threshold(Gt.dot(ys[i]) - 0.3*d)
        hs.append(h)
    return hs

def encoder(d, Gs, ys):
    l = len(Gs)
    ysc = ys
    for i in xrange(l):
        hsc = encode_one(d, Gs[i], ysc)
        ysc = hsc
    return ysc

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
        y = hs[Ni].copy()
        for li in reversed(range(l)):
            G = Gs[li]
            #h = np.sign(np.clip(G.dot(h), 0, 1))
            y = threshold(G.dot(y))
        ys.append(y)

    return ys


def compute_error(hs, ys, hsp, ysp):
    N = len(hs)
    ey, eh = 0, 0
    for i in xrange(N):
        print sum(hs[i]), sum(hsp[i]), np.linalg.norm(hs[i] - hsp[i], 1)
        print sum(ys[i]), sum(ysp[i]), np.linalg.norm(ys[i] - ysp[i], 1)
        #print
        ey += (not np.all(ys[i] - ysp[i] == 0))
        eh += (not np.all(hs[i] - hsp[i] == 0))
    return ey/float(N)
    #print 'avg. eh: ', eh/float(N)