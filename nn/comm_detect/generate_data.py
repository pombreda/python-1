import numpy as np
import scipy as sp
import scipy.sparse as sps
import pdb

#from model import *

def generate_data(n, l, d, rho, N):
    rhoy = 0.1*(2./float(d))**l
    Y = sps.rand(N, n, rhoy).todense()
    return Y