# implements a continuous-time, branching particle filter
# Crisan, Dan, Jessica Gaines, and Terry Lyons. "Convergence of a branching particle method to the solution of the Zakai equation." SIAM Journal on Applied Mathematics 58.5 (1998): 1568-1590.

import numpy as np
import pylab.matplotlib as mp

m = 5

class sys:
    def __init__(self):
        self.dim = 1
    def drift(self, x, dt):
        return -x;
    def noise(self, x, dt):
        G = sqrt(dt)
        return np.random.normal(0, G)
    def integrate(self, x, dt):
        return x + self.drift(x, dt)*dt + self.noise(x, dt)

class particle:
    def __init__(self, x0):
        self.x = x0

def integrate(sys, particles, dt):
    for p in particles:
        p.x = sys.integrate(p.x, dt)

def predict(sys, particles, Dt):
    dt = Dt/(float)m


