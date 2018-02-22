import random
import numpy as np
from scipy import sparse
from gurobipy import *

def kMIQP(r, M, lamb, k, mipGap=0.2, timeLimit=1.0, outputFlag=False):
  """ Sover function for k-MIQP.

  This code formulates and solves the following simple MIQP model:
    maximize
      r'.x-lambda * x'.Q.x
    subject to
      xi       BINARY
      x.sum() == k

  Args:
    r: `python list` contains the prob scores.
    M: `python list` contains sparse coords tuple (x,y,v)
    lamb: `python float`
    k: `python int`
    others: Parameters of gurobi model.

  Returns:
    A `python list` with size k contains indices been selected. 
    A `python float` as maximized result.
  """
  r = np.array(r)
  n = len(r)
  if isinstance(M, list):
    Q = np.zeros([n, n])
    for i, j, v in M:
      Q[i][j] = Q[j][i] = v
  else:
    Q = M
  assert n == Q.shape[0] and n == Q.shape[1]

  m = Model("qp")
  m.setParam('OutputFlag', outputFlag)
  '''
  m.setParam('TimeLimit', timeLimit)
  m.setParam('MIPGap', mipGap)
  '''

  
  '''
  #ele
  m.Params.SimplexPricing=0
  m.Params.BranchDir=1
  m.Params.Heuristics=0
  m.Params.VarBranch=0
  m.Params.Cuts=0
  m.Params.PreQLinearize=1
  '''

  #steam
  m.Params.NormAdjust=1
  m.Params.PrePasses=8
  


  '''
  #clo
  m.Params.SimplexPricing=0
  m.Params.DegenMoves=9
  m.Params.Heuristics=0
  m.Params.MIPFocus=1
  m.Params.Cuts=0
  m.Params.FlowCoverCuts=2
  m.Params.GomoryPasses=15
  m.Params.PreQLinearize=1
  m.Params.PrePasses=5
  '''
  
  


  xx = m.addVars(n, vtype=GRB.BINARY)
  x = np.array(xx.values())
  y = np.dot(r, x) - lamb*(x.T.dot(Q).dot(x))
  m.setObjective(y, GRB.MAXIMIZE)
  m.addConstr(xx.sum() == k)
  m.optimize()

  vx = []
  for i, v in enumerate(m.getVars()):
    if v.x >= 0.9:
      vx.append(i)
  assert len(vx) == k
  return vx, y.getValue()

def _test():
  r = []
  for i in range(1000):
    r.append(random.uniform(-10, -5))
  Q = [(0,2,-9.8),(5,7,-5.6),(100,200,-6),(20,80,-7.9)]
  vx, max_res = kMIQP(r, Q, lamb=0.6, k=10)
  print(vx, max_res)

if __name__ == '__main__':
  _test()
