from utils import *

from os import path

import logging
import pickle

logging.basicConfig(level=logging.INFO, format=log_format)

dataset_file = '/data/yfcc100m/yfcc100m_dataset'
in_tag_count_file = path.join(data_dir, 'imagenet_tag_count.p')
wn_tag_count_file = path.join(data_dir, 'wordnet_tag_count.p')
user_count_file = path.join(data_dir, 'flickr_user_count.p')

min_user = 20
min_in_tag = 20
min_wn_tag = 100

def get_count(in_tag_count, wn_tag_count, user_count):
  in_tag_set = set(in_tag_count.keys())
  wn_tag_set = set(wn_tag_count.keys())
  user_set = set(user_count.keys())

  in_tag_count, wn_tag_count, user_count = {}, {}, {}

  num_post = 0
  tot_line = 0
  with open(dataset_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      tot_line += 1
      if (tot_line % 20000000) == 0:
        logging.info('line#%09d %d' % (tot_line, num_post))

      fields = line.strip().split(sep_field)
      assert len(fields) == num_field
      if fields[idx_marker] != '0': # not image
        continue
      if len(fields[idx_tag]) == 0: # no tags
        continue
      user = fields[idx_user]
      if user not in user_set:
        continue
      in_tags, wn_tags = [], []
      for tag in fields[idx_tag].split(sep_tag):
        if tag in in_tag_set:
          in_tags.append(tag)
        if tag in wn_tag_set:
          wn_tags.append(tag)
      if len(in_tags) + len(wn_tags) == 0:
        continue
      title = fields[idx_title]
      description = fields[idx_description]
      is_valid = True
      for tag in in_tags + wn_tags:
        if tag not in title and tag not in description:
          is_valid = False
          break
      if not is_valid:
        continue
      num_post += 1
      for tag in in_tags:
        in_tag_count[tag] = in_tag_count.get(tag, 0) + 1
      for tag in wn_tags:
        wn_tag_count[tag] = wn_tag_count.get(tag, 0) + 1
      user_count[user] = user_count.get(tag, 0) + 1
  num_in_tag = len(in_tag_count)
  num_wn_tag = len(wn_tag_count)
  num_user = len(user_count)
  logging.info('#imagenet=%d #wordnet=%d' % (num_in_tag, num_wn_tag))
  logging.info('#user=%d #post=%d' % (num_user, num_post))
  return in_tag_count, wn_tag_count, user_count

def main():
  in_tag_count = pickle.load(open(in_tag_count_file, 'rb'))
  wn_tag_count = pickle.load(open(wn_tag_count_file, 'rb'))
  user_count = pickle.load(open(user_count_file, 'rb'))

  num_in_tag = len(in_tag_count)
  num_wn_tag = len(wn_tag_count)
  num_user = len(user_count)
  logging.info('#imagenet=%d #wordnet=%d' % (num_in_tag, num_wn_tag))
  logging.info('#user=%d' % (num_user))

  while True:
    results = get_count(in_tag_count, wn_tag_count, user_count)
    in_tag_count, wn_tag_count, user_count = results

    if (min(in_tag_count.values()) >= min_in_tag and
        min(wn_tag_count.values()) >= min_wn_tag and
        min(user_count.values()) >= min_user):
      break

    in_tag_count_tmpfile = path.join(data_dir, 'imagenet_tag_count.t')
    wn_tag_count_tmpfile = path.join(data_dir, 'wordnet_tag_count.t')
    user_count_tmpfile = path.join(data_dir, 'flickr_user_count.t')
    pickle.dump(in_tag_count, open(in_tag_count_tmpfile, 'wb'))
    pickle.dump(wn_tag_count, open(wn_tag_count_tmpfile, 'wb'))
    pickle.dump(user_count, open(user_count_tmpfile, 'wb'))
    break

    in_tag_count = {t:c for t,c in in_tag_count.items() if c >= min_in_tag}
    wn_tag_count = {t:c for t,c in wn_tag_count.items() if c >= min_wn_tag}
    user_count = {u:c for u,c in user_count.items() if c >= min_user}

if __name__ == '__main__':
  main()


