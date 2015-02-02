import numpy as np

m = 500     # y = mx1
n = 100     # x = nx1
N = 500000    # num-samples

# Y = AX
# Y = mxN, X = nxN
A = np.random.random((m,n))
mu = np.zeros((n,))
var = np.eye(n)
X = np.random.multivariate_normal(mu, var, N).T

Y = A.dot(X)

Ap = Y.dot(X.T)/float(N)
print np.linalg.norm(A-Ap, 'fro')
