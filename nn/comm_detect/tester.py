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
    rho = 0.01
    N = 5*int(np.log(n)/rho)
    '''

    n, l, d, rho, N = 50, 1, 2, 0.1, 120
    print n, l, d, rho, N
    #pdb.set_trace()

    G, H, Y = generate_nn_data(n, l, d, rho, N)
    
    Hp = encoder(d, G, Y)
    Yp = decoder(G, Hp)
    print 'autoencoder: avg. error %.4f' % (error(Y, Yp))

def test_leaner():
    n = 1000
    l = 1
    d = int(np.ceil(n**(0.15)))
    rho = 0.01
    N = 5*int(np.log(n)/rho)
    
    #n, l, d, rho, N = 10, 1, 4, 0.2, 10
    print n, l, d, rho, N
    raw_input()
    G, H, Y = generate_nn_data(n, l, d, rho, N)
    
    #G = learner(n,l,d,rho,Y)
    #print 'Learned the network'

    Hp = encoder(d, G, Y)
    Yp = decoder(G, Hp)
    print 'learner: avg. error %.4f' % (error(Y, Yp))    

def main():
    np.random.seed(42)
    test_denoising_autoencoder()

if __name__=='__main__':
    main()