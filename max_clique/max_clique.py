import networkx as nx
import numpy as np

G = nx.cycle_graph(2)
Gi = G
for i in xrange(1, 10):
    Gi = nx.tensor_product(Gi,G)
    alpha = len(nx.maximal_independent_set(Gi))
    theta = alpha**(1/float(i+1))
    print i, alpha, theta
