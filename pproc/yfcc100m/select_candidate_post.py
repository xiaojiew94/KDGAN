from utils import *

from os import path

import logging
import pickle

logging.basicConfig(level=logging.INFO, format=log_format)

dataset_file = '/data/yfcc100m/yfcc100m_dataset'
in_tag_count_file = path.join(data_dir, 'imagenet_tag_count.p')
wn_tag_count_file = path.join(data_dir, 'wordnet_tag_count.p')
user_count_file = path.join(data_dir, 'flickr_user_count.p')
def main():
  in_tag_count = pickle.load(open(in_tag_count_file, 'rb'))
  wn_tag_count = pickle.load(open(wn_tag_count_file, 'rb'))
  user_count = pickle.load(open(user_count_file, 'rb'))

  in_tag_set = set(in_tag_count.keys())
  wn_tag_set = set(wn_tag_count.keys())
  user_set = set(user_count.keys())
  num_in_tag = len(in_tag_set)
  num_wn_tag = len(wn_tag_set)
  num_user = len(user_set)
  print('#imagenet=%d #wordnet=%d' % (num_in_tag, num_wn_tag))
  print('#user=%d' % (num_user))

  tot_line = 0
  with open(dataset_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      tot_line += 1
      if (tot_line % 20000000) == 0:
        logging.info('line#%09d' % (tot_line))

      fields = line.strip().split(sep_field)
      assert len(fields) == num_field
      if fields[idx_marker] != '0': # not image
        continue
      if len(fields[idx_tag]) == 0: # no tags
        continue
      user = fields[idx_user]
      if user not in user_set:
        continue
      tags = []
      for tag in fields[idx_tag].split(sep_tag):
        if tag in in_tag_set or tag in wn_tag_set:
          tags.append(tag)
      if len(tags) == 0:
        continue
      title = fields[idx_title]
      description = fields[idx_description]
      is_valid = True
      for tag in tags:
        if tag not in title and tag not in description:
          is_valid = False
          break
      if not is is_valid:
        continue
      print(tags)
      input()

if __name__ == '__main__':
  main()


