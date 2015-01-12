import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray

a = np.random.randn(4,4).astype(np.float32)
agpu = gpuarray.to_gpu(a)
a_doubled = (2*agpu).get()

print a_doubled
print agpu