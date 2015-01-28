import numpy as np
import scipy as sp
import pdb

import cPickle as pickle
import os.path
import gzip

from model import *
from generate_data import *
from learner import *
from utils import *

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
        Gt, Ht, Yt = generate_nn_test_data(n, l, d, rho, G, N)

        training_data = (G, H, Y, n, l, d, rho, N)
        test_data = (Gt, Ht, Yt, n, l, d, rho, N)

        with gzip.open(fname, 'wb') as fp:
            pickle.dump( (training_data, test_data), fp)
        print 'Pickled data'
    else:
        with gzip.open(fname, 'rb') as fp:
            training_data, test_data = pickle.load(fp)
        print 'Loaded pickled data'
    return training_data, test_data

def test_positive_edges():
    C = np.array([
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 0, 1, 1, 1]
        ])
    d = 3
    gp = find_positive_edges(d, C, np.zeros((5,5)))
    
    res = 'The result should have 3 parents, i.e., three populated\n' + \
    'columns in gp with children (0,1,2), (0,2,4), (2,3,4)\n'
    print res, gp

def test_denoising_autoencoder():    
    
    training_data, test_data = create_data()
    G, H, Y, n, l, d, rho, N = training_data
    pdb.set_trace()

    Hp = encoder(d, G, Y)
    Yp = decoder(G, Hp)
    print 'autoencoder: dH %.4f' % (l1_loss(H, Hp))
    print 'autoencoder: dY %.4f' % (l1_loss(Y, Yp))

def test_leaner():

    training_data, test_data = create_data()
    G, H, Y, n, l, d, rho, N = training_data
    Gt, Ht, Yt, nt, lt, dt, rhot, Nt = test_data
    
    print n, l, d, rho, N
    l = 1
    d = 3
    rho = estimate_rho(l, d, Y)
    
    Gp, Hp = learner(n, l, d, rho, Y, G, H)
    print 'Learned the network'

    Yp = decoder(Gp, Hp)
    print 'learner: training error %.4f' % (l1_loss(Y, Yp))

    Htp = encoder(d, Gp, Yt)
    Ytp = decoder(Gp, Htp)
    print 'learner: test error %.4f' % (l1_loss(Yt, Ytp))

    '''
    # sanity check
    Hnew = np.concatenate((H, Ht), axis=0)
    Ynew = np.concatenate((Y, Yt), axis=0)
    N, n = Hnew.shape
    Gnewp, Hnewp = learner(n, l, d, rho, Ynew)
    print 'Learned the network with more data'
    Ynewp = decoder(Gnewp, Hnewp)
    print 'learner: training error with more data %.4f' % (l1_loss(Ynew, Ynewp))
    '''

def main():
    np.random.seed(42)
    #test_positive_edges()
    #test_denoising_autoencoder()
    test_leaner()

if __name__=='__main__':
    main()