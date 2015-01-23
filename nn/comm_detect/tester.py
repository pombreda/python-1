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

def test_with_uniform_data():
    np.random.seed(42)

    n, l, d, rho, N = 500, 1, 5, 0.01, 100000
    ys, hs, Gs = generate_network_and_data(n=n, l=l, d=d, rho=rho, N=N)


    Gsp, hsp = learn_network(n,l,d,rho,ys)
    
    # regenerate data
    ysp =  generate_data(hsp, Gsp)

    compute_error(hs, ys, hsp, ysp)

    #pdb.set_trace()