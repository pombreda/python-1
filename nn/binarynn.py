import numpy as np
import scipy as sp
import scipy.sparse as sps

n = 1000
d = 3
density = d/float(n*n)

X1 = sps.rand(n,n, density=density)
X2 = sps.rand(n,n, density=density)
Y = X1.dot(X2)/float(d)

X1hp = np.rint(Y.dot(Y.transpose()))
 



