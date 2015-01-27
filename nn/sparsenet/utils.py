import numpy as np

def threshold(A):
    A = A.astype(int)
    A[A < 0] = 0
    A[A > 0] = 1
    return A

def zero_one_loss(Y, Yp):
    '''
        0-1 loss
    '''
    N = Y.shape
    N = N[0]
    if len(Y.shape) > 1:
        N = N[0]
        return np.sum(threshold(np.sum(np.abs(Y - Yp), axis=1)))/float(N)
    else:
        return np.sum(threshold(np.abs(Y - Yp)))/float(N)

def l1_loss(Y, Yp):
    N = Y.shape
    N = N[0]
    if len(Y.shape) > 1:
        return np.sum(np.sum(np.abs(Y - Yp), axis=1))/float(N)
    else:
        return np.sum(threshold(np.abs(Y - Yp)))/float(N)

