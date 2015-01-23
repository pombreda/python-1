import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as spl

def sparse_rand_mat(n,d):
    xs = []
    ys = []
    for c in xrange(n):
        xs += np.random.choice(range(0,n), d, replace=0).tolist()
        ys += [c for i in xrange(d)]
    vals = np.sign(np.random.rand(len(xs)) - 0.5)
    return sps.coo_matrix((vals, (xs,ys)), shape=(n,n)).todense()

n = 1000
d = 3

X1 = sparse_rand_mat(n,d)
X2 = sparse_rand_mat(n, 2*d)
t1, t2 = sp.where(X2 > 0)
X2[:,:] = 0
X2[t1,t2] = 1
Y = X1.dot(X2)/float(d)

#print np.linalg.norm(X1.dot(X1.T)/np.sqrt(d) - np.eye(n))

X1hp = np.rint(Y.dot(Y.transpose()))
print np.linalg.norm((X1.dot(X1.T) - X1hp))



