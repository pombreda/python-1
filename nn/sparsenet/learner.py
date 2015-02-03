import numpy as np
import scipy as sp
import pdb
from multiprocessing import Pool

from model import *
from generate_data import *


def test_correlation_matrix(C, Cth, _G):
    l, n, _ = _G.shape
    assert l == 1

    n,_ = Cth.shape
    _Gp = _G*(_G > 0)
    c = 0
    for i in xrange(n):
        sG = set(np.nonzero(_Gp[0,:,i])[0].tolist())
        #pdb.set_trace()
        if not len(sG):
            continue
        sC = set(np.nonzero(Cth[list(sG)[0]])[0].tolist())
        sC = sC.union([i])
        if not sG.issubset(sC):
            #print 'Found weird correlation'
            #pdb.set_trace()
            #return False
            c += 1
    print 'Found weird correlations: %d, out of %d.' % (c, n)

def create_correlation_matrix(rho, Y, _G):
    N, n = Y.shape
    C = Y.T.dot(Y)
    Cth = threshold(C*(C > 2*rho*N/3.))
    #test_correlation_matrix(C, Cth, _G)
    #pdb.set_trace()
    return Cth

def get_siblings(M, s):
    siblings = set(np.where(M[s,:] > 0)[0].tolist())    # all siblings of u, say v
    return siblings

def test_and_permute_positive_edges(C, gplus, _G):
    l, n, _ = _G.shape
    assert l == 1

    _Gplus = _G*(_G > 0)
    gplus_sets = []
    _Gplus_sets = []
    gplus_new = np.zeros((n,n))
    for i in xrange(n):
        gplus_sets.append( set(np.nonzero(gplus[:,i])[0].tolist()) )
        _Gplus_sets.append( set(np.nonzero(_Gplus[0,:,i])[0].tolist()) )

    for i in xrange(n):
        toret = False
        g2 = _Gplus_sets[i]
        for j in xrange(n):
            g1 = gplus_sets[j]
            
            if (g1 <= g2) and (g2 <= g1):
                toret = True
                print 'gplus(%d) matches _Gplus(%d)' %(j, i)
                print gplus_sets[j], _Gplus_sets[i]
                gplus_new[list(gplus_sets[j]), i] = 1
                #raw_input()
                break
        if not toret:
            print 'one column wrong'
            pdb.set_trace()
    return gplus_new


def find_positive_edges(d, C, _G):
    E = C.copy()
    n, _ = C.shape
    gplus = np.zeros((n,n))
    hcounter = 0
    
    _Gplus = _G*(_G > 0)
    _Gmnius = _G*(_G < 0)

    while np.sum(E) > 0:
        Er, Ec = np.nonzero(E)
        Eri = np.random.choice(len(Er))
        print 'current parents', len(Er)
        
        v1 = Er[Eri]
        v2 = Ec[Eri]
        #print 'num edges: ', len(Er)
        #pdb.set_trace()

        Sv1 = get_siblings(C, v1)
        Sv2 = get_siblings(C, v2)
        S = Sv1.intersection(Sv2)
        
        if len(Er) > -25:
            print v1, Sv1
            print v2, Sv2
            key = raw_input()
            if key == 'd':
                pdb.set_trace()
        
        if (len(S) > 0) and (len(S) <= 1.5*d):
            Fhz = []
            #print 'reached inside'
            for v in S:
                Gammav = get_siblings(C, v)
                if len(Gammav.intersection(S)) >= 0.8*d -1:
                    Fhz.append(v)
            
            lFhz = list(Fhz)
            #print 'lFhz', lFhz
            positive_edges = set([])
            if len(lFhz):
                positive_edges = lFhz
            elif len(S) == -1:
                positive_edges = list(S)
            
            if len(positive_edges):
                gplus[positive_edges, hcounter] = 1
                hcounter += 1
                for v in positive_edges:
                    E[positive_edges, v] = 0

    print 'gplus has %d edges, Gplus has %d' % \
        (np.sum(np.abs(gplus)), np.sum(np.abs(_Gplus)))
    #gplus = test_and_permute_positive_edges(C, gplus, _G)
    return gplus
    

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
        ones_in_y = np.nonzero(y)[0].tolist()
        supph = set(np.nonzero(h)[0].tolist())
        
        '''
        # diagnostics
        print i, ones_in_y
        print i, supph
        key = raw_input()
        if key == 'd':
            pdb.set_trace()
        '''
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
        C = create_correlation_matrix(rho=rhoi, Y=Yc, _G=_G)
        gplus = find_positive_edges(d, C, _G=_G)
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
    H = threshold(g.T.dot(Y.T) - 0.3*d)
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