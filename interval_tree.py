import math
import random

class interval:
    def __init__(self, b, e):
        _b = min(b,e)
        _e = max(b,e)
        self.begin = _b
        self.end = _e
    def __repr__(self):
        return '[%d, %d]' %(self.begin, self.end)

class node:
    def __init__(self, xc, Sc, left, right):
        self.xc = xc
        self.Sc = Sc
        self.left = left
        self.right = right

class interval_tree:
    def __init__(self, intervals):
        self.root = self.divide(intervals)
    
    def divide(self, intervals):
        if not intervals:
            return None
        
        xc = self.center(intervals)
        Sl, Sc, Sr = [],[],[]

        for i in intervals:
            if i.end < xc:
                Sl.append(i)
            elif i.begin > xc:
                Sr.append(i)
            else:
                Sc.append(i)
        
        return node(xc, Sc, self.divide(Sl), self.divide(Sr))

    def _sort_by_begin(self, intervals):
        return sorted(intervals, key=lambda x: x.begin)

    def center(self, intervals):
        fs = self._sort_by_begin(intervals)
        length = len(fs)
        return fs[length/2].begin
    
    def search(self, start, end=None):
        if not end:
            return self.point_query(self.root, start, [])
        else:
            res = []
            if not (type(start) == type(1) and
                    type(end) == type(1)):
                print 'Err: can only search for integer intervals, rounding'
                start = int(math.floor(start))
                end = int(math.ceil(end))

            for j in xrange(start, end+1):
                for k in self.search(j):
                    res.append(k)
            return list(set(self._sort_by_begin(res)))
    
    def point_query(self, node, p, result):
        for k in node.Sc:
            if (k.begin <= p) and (p <= k.end):
                result.append(k)
        if p < node.xc and node.left:
            self.point_query(node.left, p, result)
        if p > node.xc and node.right:
            self.point_query(node.right, p, result)
        return list(set(result))

def basic_test():
    a = interval(1,2)
    b = interval(1,8)
    c = interval(4,7)
    tree = interval_tree([a,b,c])
    tree.search(3)

def stress_test():
    N = 100
    intervals = []
    for i in xrange(10000):
        n1, n2 = random.random()*N, random.random()*N
        intervals.append(interval(n1,n2))

    tree = interval_tree(intervals)
    for i in xrange(10):
        n1, n2 = random.random()*N, random.random()*N
        tree.search(min(n1,n2), max(n1,n2))

def main():
    basic_test()

if __name__=="__main__":
    main()
