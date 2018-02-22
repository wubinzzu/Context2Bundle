import os
import pickle
import random

import numpy as np
from tqdm import tqdm
import tensorflow as tf

class DataInput:

  def __init__(self, batch_size, raw_data, bundle_map, pad):
    self.batch_size = batch_size
    self.raw_data = raw_data
    self.epoch_size = len(raw_data) // batch_size
    if self.epoch_size * batch_size < len(raw_data):
      self.epoch_size += 1
    self.i = 0
    self.bundle_map = bundle_map
    self.pad = pad
    random.shuffle(self.raw_data)

  def __iter__(self):
    return self

  def __next__(self):
    if self.i == self.epoch_size:
      raise StopIteration

    start = self.i * self.batch_size
    end = min((self.i+1) * self.batch_size, len(self.raw_data))

    max_sl, h_sl = 0, 0
    h, i, j, nh, ni, nj = [], [], [], [], [], []
    for t in self.raw_data[start:end]:
      h.append([])
      for tt in t[1]: h[-1].extend(self.bundle_map[tt])
      i.append(list(self.bundle_map[t[2]]))
      j.append(list(self.bundle_map[t[3]]))
      ni.append(len(i[-1]))
      nj.append(len(j[-1]))
      nh.append(len(h[-1]))
      assert(ni[-1] != 0 and nj[-1] != 0)
      max_sl = max(max_sl, len(i[-1]), len(j[-1]))
      h_sl = max(h_sl, len(h[-1]))
    for k in range(end-start):
      while len(i[k]) < max_sl: i[k].append(self.pad)
      while len(j[k]) < max_sl: j[k].append(self.pad)
      while len(h[k]) < h_sl: h[k].append(self.pad)

    h = np.array(h, np.int32)
    i = np.array(i, np.int32)
    j = np.array(j, np.int32)
    nh = np.array(nh, np.int32)
    ni = np.array(ni, np.int32)
    nj = np.array(nj, np.int32)
    self.i += 1
    return self.i, (h, i, j, nh, ni, nj)


def cnn_net(name, h_emb, sl, filter_sizes, num_filters, embedding_size, dropout_rate, is_training, reuse):
  mask = tf.sequence_mask(sl, tf.shape(h_emb)[1], dtype=tf.float32) # [B, T]
  h_emb *= mask[..., None]
  # h_emb: [B, T, H]
  pad_size = tf.maximum(0, filter_sizes[-1]-tf.shape(h_emb)[1])
  h_emb = tf.pad(h_emb, [[0, 0], [0, pad_size], [0, 0]])
  h_emb = tf.expand_dims(h_emb, -1) # [B, T, H, 1]

  pooled_outputs = []
  with tf.variable_scope(name + '-cnn', reuse=reuse):
    for filter_size, num_filter in zip(filter_sizes, num_filters):
      # Convolution Layer  
      filter_shape = [filter_size, embedding_size, 1, num_filter]
      W = tf.get_variable('kernel-%d' % filter_size, shape=filter_shape)
      b = tf.get_variable('bias-%d' % filter_size, shape=[num_filter])
      conv = tf.nn.conv2d(
          h_emb,
          W,  
          strides=[1, 1, 1, 1],  
          padding="VALID",  
          name="conv")  
      # Apply nonlinearity  
      h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  
      # Maxpooling over the outputs  
      pooled = tf.reduce_max(h, axis=1, keep_dims=True)
      pooled_outputs.append(pooled)

    num_filters_total = sum(num_filters)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    h_pool_dropout = tf.layers.dropout(h_pool_flat, dropout_rate, training=is_training)
  return h_pool_dropout

class Model():

  def __init__(self, sess, rank, n_users, n_items, n_cates, bundle_rank, cate_list, reg_rate=0.0025, learning_rate = 0.1):
    self._sess = sess
    self._reg_rate = reg_rate
    self._learning_rate = learning_rate

    self.h = h = tf.placeholder(tf.int32, [None, None])
    self.i = i = tf.placeholder(tf.int32, [None, None])
    self.j = j = tf.placeholder(tf.int32, [None, None])
    self.nh = nh = tf.placeholder(tf.int32, [None])
    self.ni = ni = tf.placeholder(tf.int32, [None])
    self.nj = nj = tf.placeholder(tf.int32, [None])
    self.is_training = tf.placeholder(tf.bool, [])

    H1 = tf.get_variable('H1', [n_items, rank//2])
    C1 = tf.get_variable('C1', [n_cates, rank//2])
    B1 = tf.get_variable('B1', [n_items])
    def item_emb(id):
      return tf.concat([
        tf.nn.embedding_lookup(H1, id),
        tf.nn.embedding_lookup(C1, tf.gather(cate_list, id)),
        ], axis=-1)

    u_hist = item_emb(h)
    i_item = item_emb(i)
    j_item = item_emb(j)
    u_bias = tf.gather(B1, h)
    i_bias = tf.gather(B1, i)
    j_bias = tf.gather(B1, j)

    filter_sizes = [1, 2, 4, 8, 12, 16, 32, 64]
    num_filters = [12] * len(filter_sizes)
    u_emb = cnn_net('uhist', u_hist, nh, filter_sizes, num_filters, rank, 0.0, self.is_training, False)
    u_emb = tf.layers.dense(u_emb, rank, None)

    item_dense = tf.layers.Dense(rank, None)
    filter_sizes = [1, 2, 3, 4, 5, 6, 8, 12, 16]
    num_filters = [12] * len(filter_sizes)
    i_emb = cnn_net('item', i_item, ni, filter_sizes, num_filters, rank, 0.0, self.is_training, False)
    i_emb = item_dense(i_emb)
    j_emb = cnn_net('item', j_item, nj, filter_sizes, num_filters, rank, 0.0, self.is_training, True)
    j_emb = item_dense(j_emb)

    self.logits = tf.reduce_mean(i_bias, 1) + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1)
    x = tf.reduce_mean(i_bias - j_bias, 1) + tf.reduce_sum(tf.multiply(u_emb, i_emb) - tf.multiply(u_emb, j_emb), 1)
    self.pair_logits = x
    self.loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
          logits=x,
          labels=tf.ones_like(x)))
    self.loss += reg_rate * tf.add_n([
      tf.nn.l2_loss(u_emb),
      tf.nn.l2_loss(i_emb),
      tf.nn.l2_loss(j_emb),
      ])

    self._global_step = tf.Variable(0, trainable=False, name='global_step')
    opt = tf.train.GradientDescentOptimizer(self._learning_rate)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    self.train_op = opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self._global_step)

    self._sess.run(tf.global_variables_initializer())

  def train(self, uij):
    cost, _ = self._sess.run([self.loss, self.train_op], feed_dict={
      self.h: uij[0],
      self.i: uij[1],
      self.j: uij[2],
      self.nh: uij[3],
      self.ni: uij[4],
      self.nj: uij[5],
      self.is_training: True,
      })
    return cost

  def eval(self, uij):
    logits = self._sess.run(self.pair_logits, feed_dict={
      self.h: uij[0],
      self.i: uij[1],
      self.j: uij[2],
      self.nh: uij[3],
      self.ni: uij[4],
      self.nj: uij[5],
      self.is_training: False,
      })
    return np.mean(logits > 0)

  def predict(self, uij):
    logits = self._sess.run(self.logits, feed_dict={
      self.h: uij[0],
      self.i: uij[1],
      self.nh: uij[2],
      self.ni: uij[3],
      self.is_training: False,
      })
    return logits


  def save(self, path):
    saver = tf.train.Saver()
    save_path = saver.save(self._sess, save_path=path, global_step=self._global_step)
    print('model saved at %s' % save_path)

  def restore(self, path):
    saver = tf.train.Saver()
    saver.restore(self._sess, save_path=path)
    print('model restored from %s' % path)

  def inference(self, hist_item, uij):
    logits, pair_logits = self._sess.run([self.logits, self.pair_logits], feed_dict={
      self.h: [hist_item],
      self.i: uij[0],
      self.j: uij[1],
      self.nh: [len(hist_item)],
      self.ni: uij[2],
      self.nj: uij[3],
      self.is_training: False,
      })
    return logits, pair_logits

  
def eval(model, valid_data, bundle_map, item_count):
  auc = 0.0
  for _, uij in DataInput(64, valid_data, bundle_map, pad=item_count):
    auc += model.eval(uij) * len(uij[0])
  auc /= len(valid_data)
  return auc


def t_cnn(flag, batch_size, hidden_size, lr):
  print("flag:%r\tbatch_size:%r\thidden_size:%r\tlr:%r" %(flag, batch_size, hidden_size, lr))

  if flag == 'clo':
    source_path = '../data/bundle_clo.pkl'
    save_path = 'clo/model'
  elif flag == 'ele':
    source_path = '../data/bundle_ele.pkl'
    save_path = 'ele/model'
  else:
    assert False

  with open(source_path, 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    cate_list = pickle.load(f)
    bundle_map = pickle.load(f)
    (user_count, item_count, cate_count, bundle_count, bundle_rank, _) = pickle.load(f)

    cate_list = np.concatenate([cate_list, [cate_count]])

  res = 0.0
  tot_loss = 0.0
  gpu_options = tf.GPUOptions(allow_growth=True)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    model = Model(sess, hidden_size, user_count, item_count, cate_count, bundle_rank, cate_list, learning_rate=lr)
    for v in tf.trainable_variables():
      print(v)
    for epoch in range(50):

      auc = eval(model, test_set, bundle_map, item_count)
      print('Epoch:%d AUC:%.4f' % (epoch, auc), flush=True)
      if epoch > 0 and auc > res:
        res = auc
        model.save(save_path)

      for iter, uij in DataInput(batch_size, train_set, bundle_map, pad=item_count):
        tot_loss += model.train(uij)
        if iter % 100 == 0:
          print('Epoch:%d Iter:%d Cost:%.4f' % (epoch, iter, tot_loss/100), flush=True)
          tot_loss = 0.0

    print(res)

if __name__=="__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  random.seed(1234)
  np.random.seed(1234)
  tf.set_random_seed(1234)

  # batch_size + hidden_size + lr
  t_cnn('clo', 32, 32, 0.1)

  # 0.1 1.0
