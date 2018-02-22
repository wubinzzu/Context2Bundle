import os
import sys
sys.path.append('../utils')
import time
import pickle
import random

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from metrics import print_all_metrics

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

import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper
import masked_beam_search
from tensorflow.python.util import nest

def build_decoder_cell(rank, u_emb, batch_size, depth=2):
  cell = []
  for i in range(depth):
    if i == 0:
      cell.append(LSTMCell(rank, state_is_tuple=True))
    else:
      cell.append(ResidualWrapper(LSTMCell(rank, state_is_tuple=True)))
  initial_state = LSTMStateTuple(tf.zeros_like(u_emb), u_emb)
  initial_state = [initial_state, ]
  for i in range(1, depth):
    initial_state.append(cell[i].zero_state(batch_size, tf.float32))
  return MultiRNNCell(cell), tuple(initial_state)

from feature_aware_softmax import Dense

class Model():

  def __init__(self, sess, rank, n_users, n_items, n_cates, bundle_rank, cate_list, reg_rate=0.0025, learning_rate = 1.0):
    self._sess = sess
    self._reg_rate = reg_rate
    self._learning_rate = learning_rate

    self.h = h = tf.placeholder(tf.int32, [None, None])
    self.nh = nh = tf.placeholder(tf.int32, [None])
    self.is_training = tf.placeholder(tf.bool, [])
    batch_size = tf.shape(h)[0]

    H1 = tf.get_variable('H1', [n_items+2, rank//2])
    C1 = tf.get_variable('C1', [n_cates+1, rank//2])
    B1 = tf.get_variable('B1', [n_items+2])
    def item_emb(id):
      return tf.concat([
        tf.nn.embedding_lookup(H1, id),
        tf.nn.embedding_lookup(C1, tf.gather(cate_list, id)),
        ], axis=-1)

    u_hist = item_emb(h)
    filter_sizes = [1, 2, 4, 8, 12, 16, 32, 64]
    num_filters = [12] * len(filter_sizes)
    u_emb = cnn_net('uhist', u_hist, nh, filter_sizes, num_filters, rank, 0.0, self.is_training, False)
    u_emb = tf.layers.dense(u_emb, rank, None)
    u_emb *= tf.ones([batch_size, rank])

    go_symb = tf.ones([batch_size], tf.int32) * n_items
    eof_symb = n_items + 1

    with tf.variable_scope('seq-rnn'):
      output_layer = Dense(n_items+2, n_cates+1, cate_list, activation=None, name='output_projection')

      cell, initial_state = build_decoder_cell(rank, u_emb, batch_size)
      beam_width = 50
      initial_state = nest.map_structure(
          lambda s: seq2seq.tile_batch(s, beam_width), initial_state)
      inference_decoder = masked_beam_search.BeamSearchDecoder(
          cell=cell,
          embedding=item_emb,
          start_tokens=go_symb,
          end_token=eof_symb,
          initial_state=initial_state,
          beam_width=beam_width,
          output_layer=output_layer)

      max_decode_step = 5
      outputs, _, _ = seq2seq.dynamic_decode(
          decoder=inference_decoder,
          output_time_major=False,
          maximum_iterations=max_decode_step)
    self.scores = outputs.beam_search_decoder_output.scores[:, -1, :]
    self.outputs = tf.transpose(outputs.predicted_ids, [0, 2, 1]) # [B, W, T]

    self._sess.run(tf.global_variables_initializer())

  def inference(self, hist_item, fatch_score=False):
    outputs = self.outputs
    if fatch_score:
      outputs = [self.outputs, self.scores]
    return self._sess.run(outputs, feed_dict={
      self.h: [hist_item],
      self.nh: [len(hist_item)],
      self.is_training: False,
      })

  def restore(self, path):
    saver = tf.train.Saver()
    saver.restore(self._sess, save_path=path)
    print('model restored from %s' % path)


def main(flag):
  if flag == 'clo':
    source_path = '../data/bundle_clo.pkl'
    model_path = 'clo/model-22428'
  elif flag == 'ele':
    source_path = '../data/bundle_ele.pkl'
    model_path = 'ele/model-116000'
  else:
    assert False

  with open(source_path, 'rb') as f:
    _ = pickle.load(f)
    test_set = pickle.load(f)
    cate_list = pickle.load(f)
    bundle_map = pickle.load(f)
    (user_count, item_count, cate_count, bundle_count, bundle_rank, _) = pickle.load(f)
    gen_groundtruth_data = pickle.load(f)

  cate_list = np.concatenate([cate_list, [cate_count, cate_count]])

  def remove_tail(pred):
    pred = pred.tolist()
    while len(pred) > 0 and (pred[-1] == item_count+1 or pred[-1] == -1):
      pred.pop()
    return pred

  gpu_options = tf.GPUOptions(allow_growth=True)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = Model(sess, 64, user_count, item_count, cate_count, bundle_rank, cate_list)

    model.restore(model_path)

    res = []
    for uid, hist, pos in tqdm(gen_groundtruth_data):
      hist_item = []
      for h in hist: hist_item.extend(list(bundle_map[h]))
      predicted_ids, scores = model.inference(hist_item, fatch_score=True)
      predicted_ids, scores = predicted_ids[0], scores[0]
      groundtruth = list(bundle_map[pos])
      preds = [remove_tail(predicted_ids[j]) for j in range(predicted_ids.shape[0])]
      score = scores.tolist()
      res.append((groundtruth, preds, scores))

    print_all_metrics(flag, res, 10)

    with open('res_%s_50.pkl' % flag, 'wb') as f:
      pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  random.seed(1234)
  tf.set_random_seed(1234)
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  main('clo')
  # main('ele')
