import sys
sys.path.append('../utils')
import time
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing

from metrics import jaccard, precision, jaccard_sim, print_all_metrics
# from basic import kMIQP
from greedy import kMIQP
# from gurobi import kMIQP

 
min_max_scaler = preprocessing.MinMaxScaler()
def _preprocess(one_tuple):
  groundtruth, preds, scores = one_tuple
  # scores = np.exp(scores)
  # scores = min_max_scaler.fit_transform(np.reshape(scores, [-1, 1]))
  # scores = scores.flatten()
  M = []
  for i in range(len(preds)):
    for j in range(i+1, len(preds)):
      M.append((i, j, jaccard_sim(preds[i], preds[j])))
  return scores, M


def _watch_log(res):
  for one_tuple in res:
    r, M = _preprocess(one_tuple)
    vx, max_res = kMIQP(r, M, lamb=1.0, k=10, outputFlag=True)
    print(vx, max_res)
    break


def _watch_time_cost(res):
  start_time = time.time()
  T = 20
  for one_tuple in res[:T]:
    r, M = _preprocess(one_tuple)
    vx, max_res = kMIQP(r, M, lamb=1.0, k=10)
    print(vx, max_res)
  cost_time = time.time() - start_time
  print('User Per Second: %.4fs' % (cost_time / T))


def _watch_converge(res):
  random.shuffle(res)
  tqdmInput = tqdm(res, ncols=77, leave=True)
  prec, jacc = 0.0, 0.0
  for iter, one_tuple in enumerate(tqdmInput):
    r, M = _preprocess(one_tuple)
    vx, max_res = kMIQP(r, M, lamb=1.0, k=10)

    groundtruth, preds, scores = one_tuple
    preds = [preds[x] for x in vx]

    prec += precision(groundtruth, preds)
    jacc += jaccard(preds)

    tqdmInput.set_description('Prec@10: %.3f%% Div: %.3f'
        % (prec*100/(iter+1), jacc/(iter+1)))


def reduce_by_kMIQP(res, source_file, save_path=None):
  outputs = []
  k = 10
  for one_tuple in tqdm(res, ncols=77):
    r, M = _preprocess(one_tuple)
    vx, max_res = kMIQP(r, M, lamb=1.0, k=k)
    groundtruth, preds, scores = one_tuple
    preds = [preds[x] for x in vx]
    outputs.append((groundtruth, preds, max_res))
  if save_path is None:
    print_all_metrics(source_file, outputs, k)
  else:
    with open(save_path, 'wb') as f:
      pickle.dump(outputs, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  random.seed(1234)
  np.random.seed(1234)

  assert len(sys.argv) == 2
  source_file = sys.argv[1]
  target_file = None
  with open(source_file, 'rb') as f:
    res = pickle.load(f)
 
  # _watch_log(res)
  # _watch_time_cost(res)
  # _watch_converge(res)
  reduce_by_kMIQP(res, source_file, target_file)
