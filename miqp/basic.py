import numpy as np

def kMIQP(r, M, lamb, k):
  idx = np.argsort(r)[::-1]
  idx = idx[:k]
  max_res = sum([r[i] for i in idx])
  return idx, max_res

def _test():
  vx, max_res = kMIQP([2, 3, 1, 2.5], 0, 0, 2)
  assert len(vx) == 2
  assert vx[0] == 1 and vx[1] == 3
  assert max_res == 5.5

if __name__ == '__main__':
  _test()
