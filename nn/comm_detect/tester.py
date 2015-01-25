import numpy as np
import scipy as sp
import pdb

from model import *
from generate_data import *
from learner import *


def test_denoising_autoencoder():
    '''
    n = 1000
    l = 1
    d = int(np.ceil(n**(0.15)))
    rho = 0.1*(2./float(d))**l
    N = 10*int(np.log(n)/rho**2)
    '''
    n, l, d, rho, N = 100, 1, 2, 0.1, 100
    print n, l, d, rho, N
    raw_input()
    Y = generate_data(n, l, d, rho, N)
    G = signed_nn(n, l, d)
    pdb.set_trace()

    hsp = encoder(d, G, Y)
    ysp = decoder(hsp, Gs)
    #pdb.set_trace()
    print 'autoencoder error: %.4f' % (compute_error(hs, ys, hsp, ysp))

def main():
    np.random.seed(42)
    #test_with_uniform_data()
    test_denoising_autoencoder()

if __name__=='__main__':
    main()