import random
import numpy as np

def kMIQP(r, M, lamb, k):
  n=len(r)
  if isinstance(M, list):
    Q = np.zeros([n, n])
    for i, j, v in M:
      Q[i][j] = Q[j][i] = v
  else:
    Q = M
  assert n == Q.shape[0] and n == Q.shape[1]
  cost=[]
  mst=[]
  for i in range(n):
    cost.append(0)
    
  idx = np.argsort(r)[::-1]
  s=idx[0]
  res=r[s]
  
  mst.append(s)
  for i in range(n):
    if i!=s:
      cost[i]=r[i]-lamb*Q[s][i]
    else:
      cost[i]=np.NINF
      
  for j in range(1,k):
    maxim=np.NINF
    tmp=s
    for i in range(n):
      if cost[i]>maxim:
        maxim=cost[i]
        tmp=i
    res+=maxim
    cost[tmp]=np.NINF
    mst.append(tmp)

    for i in range(n):
      if cost[i] != np.NINF:
        cost[i] += -lamb*Q[tmp][i]

  return mst,res


def _test():
  r = []
  for i in range(1000):
    r.append(random.uniform(-10, -5))
  Q = [(0,2,-9.8),(5,7,-5.6),(100,200,-6),(20,80,-7.9)]
  vx, max_res = kMIQP(r, Q, lamb=0.6, k=10)
  print(vx, max_res)

if __name__ == '__main__':
  _test()
