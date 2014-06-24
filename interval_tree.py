class interval:
    def __init__(self, b, e):
        self.begin = b
        self.end = e
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
            for j in xrange(start, end+1):
                for k in self.search(j):
                    res.append(k)
            return _sort_by_begin(res)
    
    def point_query(self, node, p, result):
        for k in node.Sc:
            if (k.begin <= p) and (p <= k.end):
                result.append(k)
        if p < node.xc and node.left:
            self.point_query(node.left, p, result)
        if p > node.xc and node.right:
            self.point_query(node.right, p, result)
        return list(set(result))

if __name__=="__main__":
    a = interval(1, 2)
    b = interval(4, 7)
    c = interval(1, 8)
    
    tree = interval_tree([a,b,c])
    print tree.search(3)
