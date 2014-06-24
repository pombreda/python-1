from multiprocessing import Pool
import numpy as np
import time

def monte_carlo_pi(n):
    c = 0
    for i in xrange(n):
        x,y = np.random.random(), np.random.random()
        if x*x + y*y <= 1:
            c += 1
    return c

if __name__=="__main__":
    ncpu = 4
    n = int(1e7)
    
    pc = [n/ncpu for i in xrange(ncpu)]

    pool = Pool(processes=ncpu)
    
    tic = time.time()
    c = pool.map(monte_carlo_pi, pc)
    toc = time.time()
    print "pi:", sum(c)/(n*1.0)*4
    print "time:", toc-tic
