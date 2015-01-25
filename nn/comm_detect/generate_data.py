import numpy as np
import scipy as sp
import scipy.sparse as sps
import pdb

from model import *
from learner import *

def generate_nn_data(n, l, d, rho, N):
    H = generate_Y(n, 0, d, rho, N)
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
    Y = sps.rand(5*N, n, rhoy).todense()
    #pdb.set_trace()
    Y = threshold(Y)
    Y = np.array(Y)
    mask = np.all(np.equal(Y, 0), axis=1)
    
    Y = Y[~mask]
    return Y[0:N,:]

def estimate_rho(l, d, Y):
    N, n = Y.shape
    rhoy = np.sum(Y)/float(n*N)
    return rhoy*(2./float(d))**l