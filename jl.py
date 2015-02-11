import numpy as np
import scipy.spatial as spspa

def dispersion(X):
    Y = spspa.distance.pdist(X.T, 'euclidean')
    return max(Y)


d = 3000
N = 10000
eps = 0.2
m = int(np.log(N)/eps**2/np.log(1./eps))
print d, eps, N, m
raw_input()

X = np.random.random((d, N))
A = np.random.randn(m,d)
Y = A.dot(X)

for i in xrange(N):
    for j in xrange(i):
        x = X[:,i]-X[:,j]
        y = A.dot(x)
        print i,j, np.linalg.norm(y)**2/np.linalg.norm(x)**2/float(m) - 1
