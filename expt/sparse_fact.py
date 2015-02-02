import numpy as np
from sklearn.decomposition import ProjectedGradientNMF

#X = np.array([[1,1], [2, 1], [3, 1.2], \
#    [4, 1], [5, 0.8], [6, 1]])
X = np.random.random_integers(0, 1, (5,5))

model = ProjectedGradientNMF(n_components=2, init='random',\
    random_state=0)

model.fit(X)
print model.components_