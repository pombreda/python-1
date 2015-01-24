import numpy as np
import scipy as sp
import pdb

from model import *
from generate_data import *
from learner import *

def test_nn():
    generate_network_and_data(n=10, l=5, d=5, rho=0.2, N=1)

def test_correlation():
    n, l, d, rho, N = 1000, 1, 2, 0.1, 100
    ys, hs, Gs = generate_network_and_data(n=n, l=l, d=d, rho=rho, N=N)
    Gamma = create_correlation_matrix(rho=rho, ys=ys, eps=0.1)

'''
def check_statistis(hs, ys, hsp, ysp):
    compute_error(hs, ys, hsp, ysp)
    N = len(hs)
    hsparsity, ysparsity = 0,0
    for i in xrange(N):
        hsparsity += sum(hs[i])
        ysparsity += sum(ys[i])
    print ysparsity/float(N), hsparsity/float(N)
'''

def test_with_uniform_data():

    n, l, d, rho = 200, 1, 3, 0.02
    N = int(10*np.log(n)/rho)
    ys, hs, Gs = generate_network_and_data(n=n, l=l, d=d, rho=rho, N=N, \
        which_distribution='dirichlet')
    yst, hst = generate_test_data(Gs, rho, N/2, which_distribution='dirichlet')
    print 'Finished generating data'

    l, d, rho = 1, 3, 0.04
    for d in np.arange(2, 17, 2):
        Gsp, hsp = learn_network(n,l,d,rho,ys)
        
        # regenerate training data
        ysp =  decoder(hsp, Gsp)
        
        # regenerate test data
        hstp = encoder(d, Gsp, yst)
        ystp =  decoder(hstp, Gsp)

        print 'd: ', d
        print 'training error: %.4f' % (compute_error(hs, ys, hsp, ysp))
        print 'test error: %.4f' % compute_error(hst, yst, hstp, ystp)

def test_denoising_autoencoder():
    n, l, d, rho, N = 100, 1, 3, 0.02, 10
    ys, hs, Gs = generate_network_and_data(n=n, l=l, d=d, rho=rho, N=N, \
        which_distribution='uniform')
    print 'Finished generating data'

    hsp = encoder(d, Gs, ys)
    ysp = decoder(hsp, Gs)
    #pdb.set_trace()
    print 'autoencoder error: %.4f' % (compute_error(hs, ys, hsp, ysp))

def main():
    np.random.seed(42)
    #test_with_uniform_data()
    test_denoising_autoencoder()

if __name__=='__main__':
    main()