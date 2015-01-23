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
    
def test_positive_edges():
    np.random.seed(42)

    n, l, d, rho, N = 1000, 1, 2, 0.1, 100
    ys, hs, Gs = generate_network_and_data(n=n, l=l, d=d, rho=rho, N=N)
    C = create_correlation_matrix(rho=rho, ys=ys, eps=0.1)
    G = find_positive_edges(d=d, C=C)
    pdb.set_trace()