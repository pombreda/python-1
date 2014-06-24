'''
count till n using O(log log n) space
*allowed to be off by a factor of 2
'''
import numpy as np

Xavg = 0
max_tries = 100
n = int(np.random.random()*1e5)
for iter in xrange(max_tries):
    X = 0
    for i in xrange(n):
        p = np.random.random()
        if p < 1.0/pow(2,X):
            X = X+1
    Xavg += X

X = Xavg/float(max_tries)
print 'N: ', pow(2,X)-1, ' n: ', n
