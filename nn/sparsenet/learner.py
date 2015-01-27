import numpy as np
import scipy as sp
import pdb
from multiprocessing import Pool

from model import *
from generate_data import *


def create_correlation_matrix(rho, Y):
    N, n = Y.shape
    C = Y.T.dot(Y)
    C = threshold(C*(C > 2*np.log(n)/3.))
    return C

def get_siblings(C, s):
    return set(np.where(C[s,:] > 0)[0].tolist())    # all siblings of u, say v

def find_positive_edges(d, _C):
    C = _C
    n, _ = C.shape
    G = np.zeros((n,n))
    hcounter = 0
    pdb.set_trace()
    for u in xrange(n):
        # FIX: where [0] because where returns a tuple
        Su = get_siblings(C, u)
        for v in Su:
            if not v == u:
                Sv = get_siblings(C, v)
                S = Su.intersection(Sv)
                if len(S) <= 1.3*d:
                    Fhz = []
                    for s in S:
                        GammaS = get_siblings(C, s)
                        if len(GammaS.intersection(S)) >= 0.8*d -1:
                            Fhz.append(s)
                    #print 'found common cause'
                    # create a parent to explain the cause
                    #print 'Fhz:', list(Fhz)
                    lFhz = list(Fhz)
                    G[lFhz, hcounter] = 1
                    hcounter += 1
                    # delete all correlations in Fhz
                    C[lFhz, lFhz] = 0*C[lFhz, lFhz]
                    break
    print 'gplus is sparse?: ', np.sum(np.abs(G))
    return G
    

def find_negative_edges(H, Y, gplus):
    N, n = Y.shape
    gminus = -1*np.ones((n,n))
    gminus += gplus
    #pdb.set_trace()
    backedges = [set((np.nonzero(gplus[i,:])[0]).tolist()) for i in xrange(n)]
    
    c = []
    for i in xrange(N):
        h = H[i,:]
        y = Y[i,:]
        ones_in_y = np.nonzero(y)[0]
        supph = set(np.nonzero(h)[0].tolist())

        for u in ones_in_y:
            if len(supph.intersection(backedges[u])) == 1:
                gminus[u,list(supph)] = 0
                c = c+ list(supph)

    print 'set %d entries in gminus to zero' %(len(set(c)))
    print 'gminus is sparse?: ', np.sum(np.abs(gminus))
    return gminus

def learner(n, l, d, rho, Y, _G, _H):
    Yc = Y
    H = None
    G  = []
    for i in xrange(l):
        rhoi = rho*(d/2.)**(l- (i+1))
        C = create_correlation_matrix(rho=rhoi, Y=Yc)
        gplus = find_positive_edges(d, C)
        Hp = encode(d, gplus, Yc)
        
        # diagnostics
        _Gplus = _G*(_G > 0)
        _Gmnius = _G*(_G < 0)
        _Hp = encode(d, _Gplus[0, :, :], Yc)
        pdb.set_trace()
        
        gminus = find_negative_edges(Hp, Yc, gplus)
        
        g = gplus + gminus
        G.append(g)

        Yc = Hp
        H = Hp
    return np.array(G), H

def encode(d, g, Y):
    #pdb.set_trace()
    H = threshold((g.T).dot(Y.T) - 0.2*d)
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