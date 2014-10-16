import numpy as np
from scipy import sparse
from scipy import ndimage
from scipy.fftpack import dct, idct

from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

img = ndimage.imread('stata.jpg', flatten=True)
imgx, imgy = img.shape
n = img.size
num_obs = 10000

def get_data(img, num_obs):
    X = img.ravel()
    k = np.random.randint(0,n,(num_obs,))
    k = np.sort(k)
    y = X[k]

    D = dct(np.eye(n))
    A = D[k,:]

    return y, A

y, A = get_data(img, num_obs)
print 'prepared data'

lasso = Lasso(alpha=0.0001, max_iter=10000)
lasso.fit(A, y.reshape((num_obs,)))
Xhat = idct(lasso.coef_.reshape((n,1)), axis=0)
imgl1 = Xhat.reshape(imgx, imgy)

plt.figure()
plt.subplot(121)
plt.imshow(img, cmap=plt.cm.gray, interpolation = 'nearest')
plt.axis('off')
plt.subplot(122)
plt.imshow(imgl1, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')

plt.show()
