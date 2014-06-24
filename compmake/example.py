import time
from compmake import comp, compmake_console

# A few functions representing a complex workflow
def func1(param1):
    print('func1(%s)' % param1)
    time.sleep(1) # ... which takes some time
    return param1

def func2(res1, param2):
    print('func2(%s, %s)' % (res1, param2))
    time.sleep(1) # ... which takes some time
    return res1 + param1

def draw(res2):
    print('draw(%s)' % res2)

if __name__=="__main__":
    for param1 in [1,2,3]:
        for param2 in [10,11,12]:
            # Simply use "y = comp(f, x)" whenever
            # you would have used "y = f(x)".
            res1 = comp(func1, param1)
            # You can use return values as well.
            res2 = comp(func2, res1, param2)
            comp(draw, res2)
    
    compmake_console()
