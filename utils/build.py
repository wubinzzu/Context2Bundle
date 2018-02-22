import pickle
import random
import numpy as np
import pandas as pd

from tqdm import tqdm


def to_df(file_path):
  with open(file_path, 'r') as f:
    df = {}
    i = 0
    for line in tqdm(f.readlines()):
      df[i] = eval(line)
      i += 1
    df = pd.DataFrame.from_dict(df, orient='index')
    return df

def build_map(df, col_name):
  key = sorted(df[col_name].unique().tolist())
  m = dict(zip(key, range(len(key))))
  df[col_name] = df[col_name].map(lambda x: m[x])
  return m, key


def main(flag):
  if flag == 'clo':
    reviews_path = '../data/reviews_Clothing_Shoes_and_Jewelry_5.json'
    meta_path = '../data/meta_Clothing_Shoes_and_Jewelry.json'
    save_path = '../data/bundle_clo.pkl'
  elif flag == 'ele':
    reviews_path = '../data/reviews_Electronics_5.json'
    meta_path = '../data/meta_Electronics.json'
    save_path = '../data/bundle_ele.pkl'
  else:
    assert False
  random.seed(1234)
  np.random.seed(1234)

  # ========================= Convert & Remap ================================
  print('Converting %s reviews dataframe...' % flag, flush=True)
  reviews_df = to_df(reviews_path)
  reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

  print('Converting %s meta dataframe...' % flag, flush=True)
  meta_df = to_df(meta_path)
  meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
  meta_df = meta_df.reset_index(drop=True)
  meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

  asin_map, asin_key = build_map(meta_df, 'asin')
  cate_map, cate_key = build_map(meta_df, 'categories')
  revi_map, revi_key = build_map(reviews_df, 'reviewerID')

  user_count, item_count, cate_count, example_count = \
      len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
  print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count), flush=True)

  meta_df = meta_df.sort_values('asin')
  meta_df = meta_df.reset_index(drop=True)
  reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
  reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
  reviews_df = reviews_df.reset_index(drop=True)
  reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
  
  cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
  cate_list = np.array(cate_list, dtype=np.int32)

  meta = []
  for i, m in meta_df.iterrows():
    m = dict(m)
    del m['related']
    del m['description']
    m['asin'] = asin_key[i]
    meta.append(m)
  prices = [0.0 if np.isnan(i['price']) else i['price'] for i in meta]

  # =========================== Build Dataset ================================
  print('Building bundle_map...')
  bundle_all = [(-1,)]
  for _, hist in tqdm(reviews_df.groupby('reviewerID')):
    hist_group = hist.groupby('unixReviewTime')
    if len(hist_group) <= 1: continue
    for _, group in hist_group:
      bundle = group['asin'].tolist()
      bundle.sort(key=lambda x: (-prices[x], x))
      bundle_all.append(tuple(bundle))
  bundle_all = sorted(set(bundle_all))
  bundle_map = dict(zip(bundle_all, range(len(bundle_all))))
  bundle_count = len(bundle_all) - 1
  bundle_max_len = len(bundle_all[0])

  print('Building train/valid dataset...')
  train_set, valid_set = [], []
  for uid, hist in tqdm(reviews_df.groupby('reviewerID')):
    hist_group = hist.groupby('unixReviewTime')
    if len(hist_group) <= 1: continue
    bundle_list = []
    for _, group in hist_group:
      bundle = group['asin'].tolist()
      bundle.sort(key=lambda x: (-prices[x], x))
      bundle_list.append(bundle_map[tuple(bundle)])

    def gen_neg():
      neg = bundle_list[0]
      while neg in bundle_list:
        neg = random.randint(1, bundle_count)
      return neg
    neg_list = [gen_neg() for i in range(len(bundle_list))]

    for i in range(1, len(bundle_list)):
      hist_i = bundle_list[:i]
      if i != len(bundle_list) - 1:
        train_set.append((uid, hist_i, bundle_list[i], neg_list[i]))
      else:
        valid_set.append((uid, hist_i, bundle_list[i], neg_list[i]))
  random.shuffle(train_set)
  random.shuffle(valid_set)

  # =========================== Build Gen Groundtruth ========================
  print('Building groundtruth data for generation...')
  gen_groundtruth_data = [t for t in valid_set if len(bundle_all[t[2]]) >= 2]
  idx = np.random.choice(len(gen_groundtruth_data), 1000, replace=False)
  gen_groundtruth_data = [(gen_groundtruth_data[i][0],
    gen_groundtruth_data[i][1], gen_groundtruth_data[i][2]) for i in idx]


  # ============================== Save Pickle ===============================
  with open(save_path, 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(bundle_all, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count, bundle_count, bundle_max_len, example_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(gen_groundtruth_data, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(meta, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)
  print('Finish, save to %s\n' % save_path, flush=True)


if __name__ == '__main__':
  main('clo')
  # main('ele')
