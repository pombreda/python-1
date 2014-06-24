import pprocess as pp
import time
import numpy as np

def time_killer(n):
    for i in xrange(n):
        t1 = np.std(np.random.randn(1e5))
    return t1

args = [500, 500]

serial = 0

if serial:
    tic = time.time()
    res1 = [time_killer(arg) for arg in args]
    print "%.3f [s] for serial" % (time.time() - tic)
else:
    ncpu = 2
    res = pp.Map(limit=ncpu, reuse=1)
    par_func = res.manage(pp.MakeReusable(time_killer))
    tic = time.time()
    [par_func(arg) for arg in args]
    res2 = res[0:3]
    print "%.3f [s] for parallel" % (time.time() - tic)
