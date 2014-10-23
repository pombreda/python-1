import numpy as np
from scipy import sparse
from scipy import ndimage
from scipy.fftpack import dct, idct

from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

img = ndimage.imread('statas.jpg', flatten=True)
imgx, imgy = img.shape
n = img.size
num_obs = n/10
X = img.ravel()
Xdct = dct(X)

def get_data(num_obs):
    A = np.random.normal(0,1/np.sqrt(n), (num_obs,n))
    y = np.dot(A, Xdct)
    return y, A

y, A = get_data(num_obs)
print 'prepared data'

lasso = Lasso(alpha=0.01, max_iter=1000)
lasso.fit(A, y)
Xhat = idct(lasso.coef_)
imgl1 = Xhat.reshape(imgx, imgy)

plt.figure()
plt.subplot(121)
plt.imshow(img, cmap=plt.cm.gray, interpolation = 'nearest')
plt.axis('off')
plt.subplot(122)
plt.imshow(imgl1, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')

plt.show()
