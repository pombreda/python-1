from sklearn.datasets import load_svmlight_file
import numpy as np
import scipy.sparse as sps
import pdb, gzip, sys


def save_sparse_csr(filename, array):
    np.savez(filename, data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sps.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def csr_vstack(a, b):
    a.data = np.hstack((a.data, b.data))
    a.indices = np.hstack((a.indices, b.indices))
    a.indptr = np.hstack((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0] + b.shape[0], b.shape[1])
    return a

def write():
    num_features = 3231961
    X, Y = load_svmlight_file('url_svmlight/Day0.svm')
    for i in xrange(1, 121):
        fname = 'url_svmlight/Day'+str(i)+'.svm'
        print fname
        x, y = load_svmlight_file(fname)
        #pdb.set_trace()
        X = csr_vstack(X, x)
        Y = np.array(Y.tolist() + y.tolist())

    save_sparse_csr('X.npz', X)
    np.save('Y.npy', Y)

def get_data():
    X = load_sparse_csr('X.npz')
    Y = np.load('Y.npy')
    return X,Y

if __name__=='__main__':
    write()