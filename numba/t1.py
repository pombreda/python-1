import numba
import numpy as np
from contracts import contract, new_contract

img = new_contract('img', 'array[NxN],N>0')

@numba.autojit
#@contract(a='img', b='img',
#        returns='img')
def avg_dist(a, b):
    return a+b

@numba.autojit
def main():
    a = np.random.random((5,5))
    b = np.random.random((5,5))
    avg_dist(a,b)

if __name__=='__main__':
    main()
