import numpy as np
import scipy as sp
import pdb

import cPickle as pickle
import os.path
import gzip

from model import *
from generate_data import *
from learner import *

fname = 'tester.pkl.gz'

def create_data():
    if not os.path.isfile(fname):
        
        n = 1000
        l = 1
        d = int(np.ceil(n**(0.15)))
        rho = 0.01
        N = 10*int(np.log(n)/rho)
        
        #n, l, d, rho, N = 1000, 1, 2, 0.01, 1000
        print n, l, d, rho, N
        #pdb.set_trace()

        G, H, Y = generate_nn_data(n, l, d, rho, N)
        with gzip.open(fname, 'wb') as fp:
            pickle.dump((G, H, Y, n, l, d, rho, N), fp)
        print 'Pickled data'
    else:
        with gzip.open(fname, 'rb') as fp:
            G, H, Y, n, l, d, rho, N = pickle.load(fp)
        print 'Loaded pickled data'
    return G, H, Y


def test_denoising_autoencoder():    
    
    G, H, Y = create_data()
    
    Hp = encoder(d, G, Y)
    Yp = decoder(G, Hp)
    print 'autoencoder: avg. error %.4f' % (error(Y, Yp))

def test_leaner():

    G, H, Y = create_data()
    
    n = 1000
    l = 1
    d = 3
    rho = 0.01
    Gp, Hp = learner(n, l, d, rho, Y)
    print 'Learned the network'

    #pdb.set_trace()
    #Hp = encoder(d, Gp, Y)
    Yp = decoder(Gp, Hp)
    print 'learner: avg. error %.4f' % (error(Y, Yp))    

def main():
    np.random.seed(42)
    #test_denoising_autoencoder()
    test_leaner()

if __name__=='__main__':
    main()