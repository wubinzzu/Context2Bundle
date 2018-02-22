import pickle
import sys
sys.path.append('../utils')
from metrics import print_all_metrics

flag = 'clo'

if flag == 'clo':
  source_path = 'rank_clo_all.pkl'
elif flag == 'ele':
  source_path = 'rank_ele_all.pkl'
else:
  assert False

with open(source_path, 'rb') as f:
  res = pickle.load(f)
  bundle_map = pickle.load(f)

k = 10
outputs = []
for uid, hist, pos, logits in res:
  groundtruth = list(bundle_map[pos])
  idx = logits.argsort()[::-1] + 1
  preds, scores = [], []
  i = 0
  while len(preds) < k:
    if idx[i] not in hist and len(bundle_map[idx[i]]) >= 2:
      preds.append(list(bundle_map[idx[i]]))
      scores.append(logits[idx[i]-1])
    i += 1
  outputs.append((groundtruth, preds, scores))

  idx = [i for i in idx.tolist() if i not in hist]

print_all_metrics(flag, outputs, k)
