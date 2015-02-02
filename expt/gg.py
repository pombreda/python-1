import numpy as np
import matplotlib.pyplot as plt

plt.ion()

def draw_poisson(lamda):
    num_points = np.random.poisson(lamda)
    return np.random.rand(num_points,2)

fig = plt.figure()

def draw_points(EN, ER):
    plt.clf()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    X = draw_poisson(EN)
    n,_ = X.shape
    R = ER*np.random.rand(n)
    
    for i in xrange(n):
        patch = plt.Circle((X[i,0], X[i,1]), alpha=0.5, \
            edgecolor='none', radius=R[i], facecolor='black')
        ax.add_patch(patch)
    plt.draw()

draw_points(25, 0.05)
plt.show()

#if __name__=="__main__":
#    main()