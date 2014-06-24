import numba
from math import sqrt

@numba.jit(numba.bool_(numba.int64), nopython=True)
def is_prime(n):
    d = int(sqrt(n))
    while d > 1:
        if n % d == 0:
            return False
        else:
            d = d-1
    return True
