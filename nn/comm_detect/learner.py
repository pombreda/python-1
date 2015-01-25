import numpy as np
import scipy as sp
import pdb

from model import *
from generate_data import *


def create_correlation_matrix(rho, Y):
    N, n = Y.shape

    C = np.zeros((n,n))
    for i in xrange(N):
        yt = np.atleast_2d(Y[i,:])
        C += (yt.T).dot(yt)

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
    

def find_negative_edges(H, Y, gplus):
    N, n = Y.shape
    gminus = -1*np.ones((n,n))
    gminus += gplus
    #pdb.set_trace()
    backedges = [set((np.nonzero(gplus[i,:])[0]).tolist()) for i in xrange(n)]
    
    for h, y in zip(H, Y):
        ones_in_y, _ = np.nonzero(y)
        supph = set(np.nonzero(h)[0].tolist())
        for u in ones_in_y:
            if len(supph.intersection(backedges[u])) == 1:
                gminus[u,list(supph)] = 0
    return gminus

def learner(n, l, d, rho, Y):
    Yc = Y
    H = None
    G  = []
    for i in xrange(l):
        rhoi = rho*(d/2.)**(l- (i+1))
        C = create_correlation_matrix(rho=rhoi, Y=Yc)
        gplus = find_positive_edges(d, C)
        Hp = encode(d, gplus, Yc)
        gminus = find_negative_edges(Hp, Yc, gplus)
        #pdb.set_trace()
        g = gplus + gminus
        G.append(g)

        Yc = Hp
        H = Hp
    return np.array(G), H

def encode(d, g, Y):
    N, n = Y.shape
    #pdb.set_trace()
    H = g.T.dot(Y.T) - 0.3*d
    return H.T

def encoder(d, G, Y):
    l,n,m = G.shape
    Hc = Y
    for i in xrange(l):
        g = np.squeeze(G[i,:,:])
        Hc = encode(d, g, Hc)
    return Hc

def decode(g, H):
    return threshold(g.dot(H.T)).T

def decoder(G, H):
    l,n,m = G.shape
    N, n = H.shape

    Y = H
    for li in reversed(range(l)):
        g = np.squeeze(G[li,:,:])
        Y = decode(g, Y)
    return Y

def error(Y, Yp):
    N, n = Y.shape
    pdb.set_trace()
    dY  = threshold(np.sum(np.abs(Y - Yp), axis=1))
    return np.sum(dY)/float(N)
        
