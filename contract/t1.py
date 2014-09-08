from contracts import contract, new_contract
import numpy as np

@contract(a='array[NxN],N>0',
        returns='array[NxN],N>0')
def check_contract(a):
    return a

def main():
    a = np.zeros((2,2))
    b = check_contract(a)
    print b

if __name__=="__main__":
    main()
