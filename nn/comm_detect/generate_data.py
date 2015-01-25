import numpy as np
import scipy as sp
import pdb

#from model import *

def generate_data(n, l, d, rho, N):
    rhoy = 0.1*(2./float(d))**l
    Y = sp.sparse.rand(N, n, rhoy).todense()
    return Y