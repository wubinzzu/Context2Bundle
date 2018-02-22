import sys
sys.path.append('../utils')
import pickle
import numpy as np
from tqdm import tqdm
from collections import Counter
from metrics import jaccard, precision

 
def subsets(nums, mi=2, ma=5):
  res = [[]]
  for num in sorted(nums):
    res += [item+[num] for item in res if len(item) < ma]
  res = [tuple(t) for t in res if len(t) >= mi]
  return res

def main(flag, k):

  if flag == 'clo':
    source_path = '../data/bundle_clo.pkl'
  elif flag == 'ele':
    source_path = '../data/bundle_ele.pkl'
  else:
    assert False

  with open(source_path, 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    cate_list = pickle.load(f)
    bundle_map = pickle.load(f)
    (user_count, item_count, cate_count, bundle_count, bundle_rank, _) = pickle.load(f)
    gen_groundtruth_data = pickle.load(f)

  freq = Counter()
  for t in train_set:
    if len(bundle_map[t[2]]) >= 2:
      t = bundle_map[t[2]]
      freq.update(subsets(t))
      # for i in range(len(t)):
      #   for j in range(i+1, len(t)):
      #     freq.update([tuple([t[i], t[j]])])

  preds = freq.most_common(k)
  preds = [[i for i in t[0]] for t in preds]

  total, jacc, prec = 0, jaccard(preds), 0.0
  for uid, hist, pos in gen_groundtruth_data:
    groundtruth = list(bundle_map[pos])
    prec += precision(groundtruth, preds)
    total += 1

  print(flag, 'P@%d: %.4f%%\tDiv: %.4f' % (k, prec*100/total, jacc))

if __name__ == '__main__':
  main('clo', k=5)
  main('clo', k=10)
  main('ele', k=5)
  main('ele', k=10)
