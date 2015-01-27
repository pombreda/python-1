import numpy as np
import scipy as sp
import pdb
from sklearn import svm

import cPickle as pickle
import os.path
import gzip

from model import *
from generate_data import *
from learner import *
from utils import *

fname = 'data/mnist.pkl.gz'


def test_mnist():
    train_set, valid_set, test_set = generate_mnist(fname)
    Y, target = train_set
    Yt, targett = test_set
    #pdb.set_trace()
    #Y = np.concatenate((Y, 0*Y), axis=1)
    #Yt = np.concatenate((Yt, 0*Yt), axis=1)

    N, n = Y.shape
    l = 2
    d = int(np.ceil(n**(0.19)))
    rho = estimate_rho(l, d, Y)
    #N = int(np.log(n)/rho**2)
    
    if N >= Y.shape[0]:
        print 'samples N: %.d, Y: %.d' %(N, Y.shape[0])
    else:
        N = int(np.log(n)/rho**2)
        Y = Y[0:N,:]
        target = target[0:N]
        Yt = Yt[0:N,:]
        targett = targett[0:N]
    
    print n, l, d, rho, N
    Gp, Hp = learner(n, l, d, rho, Y)
    print 'Learned the network'

    Yp = decoder(Gp, Hp)
    print 'NN training error: %.4f' % l1_loss(Y, Yp)
    
    '''
    clf = svm.LinearSVC(loss='l2', penalty='l1', dual=False)
    clf.fit(Hp, target)
    targetp = clf.predict(Hp)
    print 'training error: %.4f' % (zero_one_loss(target, targetp))

    Htp = encoder(d, Gp, Yt)
    targettp = clf.predict(Htp)
    print 'test error: %.4f' % (zero_one_loss(targett, targettp))
    '''
def main():
    test_mnist()

if __name__=='__main__':
    main()