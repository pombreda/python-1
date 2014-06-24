from numbapro import vectorize, float32

@vectorize([float32(float32, float32)],
    target='parallel')
def sum(a, b):
    return a+b
