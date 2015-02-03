import numpy as np
import scipy as sp
import scipy.sparse as sps
import pdb
import gzip


import cPickle as pickle
import os.path
import gzip

from model import *
from learner import *

def generate_nn_data(n, l, d, rho, N):
    H = generate_Y(n, 0, d, rho, N)
    N_, n_ = H.shape
    if N_ != N:
        print 'Could not generate %d samples, only %d generated' %(N, N_)
    G = signed_nn(n, l, d)
    Y = decoder(G, H)
    return G, H, Y

def generate_nn_test_data(n, l, d, rho, G, N):
    H = generate_Y(n, 0, d, rho, N)
    Y = decoder(G, H)
    return None, H, Y

def generate_Y(n, l, d, rho, N):
    rhoy = rho*(d/2.)**l
    if rhoy > 0.1:
        print 'output is not sparse, rhoy=', rhoy
    
    Y = sps.rand(N, n, rhoy).todense()
    #pdb.set_trace()
    Y = threshold(Y)
    Y = np.array(Y)
    mask = np.all(np.equal(Y, 0), axis=1)
    
    for i in np.nonzero(mask)[0]:
        p = int(n*np.random.random())
        Y[i, p] = 1
    return Y

def estimate_rho(l, d, Y):
    N, n = Y.shape
    rhoy = np.sum(Y)/float(n*N)
    return rhoy*(2./float(d))**l

'''
# static variables inside the function!
def generate_sparse_Y(n,l,d,rho,N):
    if not (hasattr(generate_sparse_Y, "signal") or \
            hasattr(generate_sparse_Y, "m") or \
            hasattr(generate_sparse_Y, "density") or \
            hasattr(generate_sparse_Y, "A")):
        m = 5*n
        signal = np.random.random((m,1))
        density = 0.2
        A = np.random.random((n,m))
    Y = []
    rhoy = rho*(d/2.)**l
    for i in xrange(N):
        M = np.random.random((n,m))
        mask = M*(M < density)
        pdb.set_trace()
        B = mask*A
        y = np.squeeze(np.array(B.dot(signal)))
        y = y/np.sum(y)
        Y.append(threshold(y))
    return np.array(Y)
'''

def generate_mnist(fname):
    train_set, valid_set, test_set = None, None, None
    with gzip.open(fname, 'rb') as fp:
        train_set, valid_set, test_set = pickle.load(fp)    

    train_set = (threshold(train_set[0]), train_set[1])
    test_set = (threshold(test_set[0]), test_set[1])
    valid_set = (threshold(valid_set[0]), valid_set[1])
    return train_set, valid_set, test_set

fname = 'data/uniform_sparse.pkl.gz'
def create_data():
    if not os.path.isfile(fname):
        
        n = 100
        l = 1
        d = int(np.ceil(n**(0.15)))
        rho = 0.01
        N = int(np.log(n)/rho**2)
        
        #n, l, d, rho, N = 10, 1, 2, 0.2, 1000
        print n, l, d, rho, N
        #pdb.set_trace()

        G, H, Y = generate_nn_data(n, l, d, rho, N)
        N, _ = Y.shape
        Gt, Ht, Yt = generate_nn_test_data(n, l, d, rho, G, N)
        print 'Finished constructing data'

        training_data = (G, H, Y, n, l, d, rho, N)
        test_data = (Gt, Ht, Yt, n, l, d, rho, N)

        fp = gzip.open(fname, 'wb')
        pickle.dump( (training_data, test_data), fp)
        print 'Pickled data'
        fp.close()
    else:
        fp = gzip.open(fname, 'rb')
        training_data, test_data = pickle.load(fp)
        fp.close()
        print 'Loaded pickled data'
    return training_data, test_data