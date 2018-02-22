import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from tqdm import tqdm, trange
import pickle

import numpy as np
import tensorflow as tf
tf.set_random_seed(1234)

from cnn import Model

flag = 'clo'

if flag == 'clo':
  source_path = '../data/bundle_clo.pkl'
  model_path = 'clo/model-16020'
  save_path = 'rank_clo_all.pkl'
  batch_size = 1200
elif flag == 'ele':
  source_path = '../data/bundle_ele.pkl'
  model_path = 'ele/model-139200'
  save_path = 'rank_ele_all.pkl'
  batch_size = 120
else:
  assert False

with open(source_path, 'rb') as f:
  _ = pickle.load(f)
  _ = pickle.load(f)
  cate_list = pickle.load(f)
  bundle_map = pickle.load(f)
  (user_count, item_count, cate_count, bundle_count, bundle_rank, _) = pickle.load(f)
  gen_groundtruth_data = pickle.load(f)

  cate_list = np.concatenate([cate_list, [cate_count, cate_count]])


class Input():

  def __init__(self, batch_size, raw_data, bundle_map, pad):
    self.batch_size = batch_size
    self.raw_data = raw_data
    self.epoch_size = len(raw_data) // batch_size
    if self.epoch_size * batch_size < len(raw_data):
      self.epoch_size += 1
    self.i = 0
    self.bundle_map = bundle_map
    self.pad = pad

  def __iter__(self):
    return self

  def __next__(self):
    if self.i == self.epoch_size:
      raise StopIteration

    start = self.i * self.batch_size
    end = min((self.i+1) * self.batch_size, len(self.raw_data))

    i, ni = [], []
    for t in self.raw_data[start:end]:
      i.append(list(self.bundle_map[t]))
      ni.append(len(i[-1]))
    max_sl = max(ni)
    for k in range(end-start):
      while len(i[k]) < max_sl: i[k].append(self.pad)

    i = np.array(i, np.int32)
    ni = np.array(ni, np.int32)
    self.i += 1
    return i, ni

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  model = Model(sess, 32, user_count, item_count, cate_count, bundle_rank, cate_list)
  model.restore(model_path)

  items, nitems = [], []
  for i, ni in Input(batch_size, list(range(1, bundle_count+1)), bundle_map, item_count+1):
    items.append(i)
    nitems.append(ni)
    # assert np.all(ni > 1)

  res = []
  # for iter, (uid, hist, pos) in enumerate(gen_groundtruth_data):
  for (uid, hist, pos) in tqdm(gen_groundtruth_data):
    hist_item = []
    for h in hist: hist_item.extend(list(bundle_map[h]))

    logits = []
    for i in range(len(items)):
      logits.append(model.predict(([hist_item], items[i], [len(hist_item)], nitems[i])))
    logits = np.concatenate(logits)

    assert logits.shape[0] == bundle_count

    groundtruth = list(bundle_map[pos])
    res.append((uid, hist, pos, logits))
    # print(iter, flush=True)

  with open(save_path, 'wb') as f:
    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(bundle_map, f, pickle.HIGHEST_PROTOCOL)
