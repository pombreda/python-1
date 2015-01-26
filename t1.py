import numpy as np

a = np.array([
    [1, 0, 0],
    [0, np.nan, 0],
    [0, 0, 0],
    [np.nan, np.nan, np.nan],
    [2, 3, 4]
])

mask = np.all(np.isnan(a) | np.equal(a, 0), axis=1)
print a[~mask]