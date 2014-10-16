import gzip, csv
import glob
import numpy as np
import matplotlib.pyplot as plt

fnames = glob.glob('data/*.csv.gz')

cmax = 1
utime,cost,vol = [],[],[]
for f in fnames:
    t,c,v = np.loadtxt(f, delimiter=',', unpack=True)
    utime += t.tolist()
    cost += c.tolist()
    vol += v.tolist()
    print f

    cmax -= 1
    if not cmax:
        break

tc = zip(utime, cost)
tc = sorted(tc, key=lambda x: x[0])
tc = [[x[0],x[1]] for x in tc if x[1] < 2000]
tc = np.array(tc)
plt.plot(tc[:,0], tc[:,1])
plt.show()
