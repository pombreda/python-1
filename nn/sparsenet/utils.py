import numpy as np

def zero_one_loss(Y, Yp):
    '''
        0-1 loss
    '''
    N, n = Y.shape
    return np.sum(threshold(np.sum(np.abs(Y - Yp), axis=1)))/float(N)

def l1_loss(Y, Yp):
    N, n = Y.shape
    return np.sum(np.sum(np.abs(Y - Yp), axis=1))/float(N)

