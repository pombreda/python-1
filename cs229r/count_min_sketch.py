import hashlib
import array

class count_min_sketch():
  '''
    d hash functions
    w image of hash function
  '''
  def __init__(self, w, d):
    if not w or not d:
      raise ValueError('w and d should be non-zero')

    self.w, self.d = w, d
    self.n = 0
    self.tables = []
    for _ in xrange(d):
      table = array.array('H', (0 for _ in xrange(w)))
      self.tables.append(table)

  def _hash(self, x):
    md5 = hashlib.md5(str(hash(x)))
    for i in xrange(self.d):
      md5.update(str(i))
      yield int(md5.hexdigest(), 16) % self.w

  def add(self,x,value=1):
    self.n += value
    for table,i in zip(self.tables, self._hash(x)):
      table[i] += value

  def query(self, x):
    return min(table[i] for table,i in zip(self.tables, self._hash(x)))

  def __getitem__(self,x):
    return self.query(x)

  def __len__(self):
    return self.n


def main():
  cms = count_min_sketch(1000, 10)
  cms.add(1, value=123)
  cms.add(3, value=23)
  cms.add(2, value=2)

  #print cms[3]

if __name__=='__main__':
  main()
