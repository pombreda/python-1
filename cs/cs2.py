import numpy as np
from scipy import sparse
from scipy import ndimage
from scipy.fftpack import dct, idct

from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


freq = 2 + np.random.random_sample(25)*15
t = np.arange(0, 100, 0.1)
X = np.array([np.sin(f*t) for f in freq])
X = np.sum(X, axis=0)

n = len(X)
num_obs = n/10

lasso = Lasso(alpha=0.001, max_iter=1000)

def get_data(num_obs):
    k = np.random.randint(0, n, (num_obs,))
    k = np.sort(k)
    y = X[k]
    D = dct(np.eye(n))
    A = D[k,:]
    
    return y, A

for num_obs_f in xrange(1, 20): 
    num_obs = int(n/float(num_obs_f))
    y, A = get_data(num_obs)
    lasso.fit(A, y)
    Xh = idct(lasso.coef_)
    err = np.linalg.norm(X-Xh)
    print num_obs, ': ', err

