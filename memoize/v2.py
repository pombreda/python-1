from math import sqrt
from joblib import delayed, Parallel, Memory
import numpy as np

memory = Memory(cachedir='/tmp/joblib', verbose=1)

@memory.cache
def f(x):
    print ('Runing f(%s)' %x)
    return x

print Parallel(n_jobs=2, backend="threading")(delayed(sqrt)(i**2) for i in xrange(10))
