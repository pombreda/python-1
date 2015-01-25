import numpy as np
import scipy as sp
import scipy.sparse as sps
import pdb

from model import *

def generate_data(n, l, d, rho, N):
    rhoy = rho*(d/2.)**l
    Y = sps.rand(N, n, rhoy).todense()
    return threshold(Y)