import os
import sys
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def jaccard_sim(a, b):
  aa = set(a)
  bb = set(b)
  aa, bb = aa&bb, aa|bb
  if len(bb) == 0: return 0.0
  return len(aa & bb) / len(aa | bb)

def jaccard(preds):
  jac = 0.0
  for i in range(len(preds)):
    for j in range(i+1, len(preds)):
      jac += jaccard_sim(preds[i], preds[j])
  jac /= len(preds) * (len(preds) - 1) / 2
  return jac

def precision(groundtruth, preds):
  prec = 0.0
  if not isinstance(groundtruth[0], list):
    groundtruth = [groundtruth]
  for grou in groundtruth:
    prec_part = 0.0
    for pred in preds:
      # prec_part = max(prec_part, jaccard_sim(pred, grou))
      prec_part += jaccard_sim(pred, grou)
    prec_part /= len(preds)
    prec += prec_part
  prec /= len(groundtruth)
  return prec

def print_all_metrics(flag, res, k=10):

  total = len(res)
  soft_prec, prec, jacc = 0.0, 0.0, 0.0
  for groundtruth, preds, scores in res:
    preds = preds[:k]

    prec += precision(groundtruth, preds)
    jacc += jaccard(preds)
  print('%s\tP@%d: %.4f%%\tDiv: %.4f'
      % (flag, k, prec*100/total, -jacc/total))


if __name__ == '__main__':
  assert len(sys.argv) == 2
  pkl_path = sys.argv[1]

  with open(pkl_path, 'rb') as f:
    res = pickle.load(f)

  k = 10
  flag = "clo"
  if 'ele' in pkl_path:
    flag = 'ele'
  print_all_metrics(flag, res, 10)
  print_all_metrics(flag, res, 5)
