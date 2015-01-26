from multiprocessing import Pool, Array
import ctypes
import numpy as np

'''
def f(x):
    for i in xrange(10):
        x = x*x
    return x

def stress_test_par(n):
    xargs = range(1, int(n))
    pool = Pool(8)
    res = pool.map(f, xargs)

def stress_test_seq(n):
    xargs = range(1, int(n))
    for x in xargs:
        f(x)

stress_test_par(100000)
#stress_test_seq(100000)
'''

n = 10000
shared_array_base = Array(ctypes.c_double, n*n)
shared_array = np.frombuffer(shared_array_base.get_obj())
shared_array = shared_array.reshape(n, n)

def f(i, def_param=shared_array):
    shared_array[i,:] = i

pool = Pool(6)
pool.map(f, range(n))
