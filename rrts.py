from pylab import *
import sys,time

random.seed(42)

class vertex_t:
    def __init__(self, z, index, c=0, cp=0):
        self.z, self.index = z, index
        self.c, self.cp = c, cp

        self.pi = -1

class opt_data_t:
    def __init__(self, param):
        self.param = param

class edge_t:
    def __init__(self, v1i=-1, v2i=-1, index=-1, opt_data):
        self.v1i, self.v2i = v1i, v2i
        self.index = index
        self.opt_data = opt_data

class rrts_t:
    def __init__(self):
        self.keys, self.vertices, self.edges = [],[],[]
        self.gamma = 2.1
        self.best_cost, self.best_vertex = 1e10, -1

        # add root
