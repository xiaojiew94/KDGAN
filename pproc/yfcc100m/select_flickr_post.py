from nltk.corpus import wordnet
from operator import itemgetter
from sys import stdout

import pickle
import string

dataset_file = '/data/yfcc100m/yfcc100m_dataset'
in_file_p = 'imagenet_tag_set.p'
wn_file_p = 'wordnet_tag_set.p'
in_file_f = 'imagenet_tag_set.f'
wn_file_f = 'wordnet_tag_set.f'
user_file_f = 'flickr_user_set.f'

num_field = 25
idx_user = 3
idx_tag = 10
idx_marker = 24
sep_field = '\t'
sep_tag = ','

min_user = 100
min_in_tag = 50
min_wn_tag = 200

def reduce(in_tag_set, wn_tag_set):
  user_count = {}
  tot_line = 0
  with open(dataset_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      tot_line += 1
      if (tot_line % 20000000) == 0:
        print('line#%09d' % (tot_line))

      fields = line.strip().split(sep_field)
      assert len(fields) == num_field
      if fields[idx_marker] != '0': # not image
        continue
      if len(fields[idx_tag]) == 0: # no tags
        continue
      is_valid = False
      tags = fields[idx_tag].split(sep_tag)
      for tag in tags:
        if tag in in_tag_set or tag in wn_tag_set:
          is_valid = True
          break
      if not is_valid:
        continue
      user = fields[idx_user]
      user_count[user] = user_count.get(user, 0) + 1
  user_set = set([u for u,c in user_count.items() if c >= min_user])

  num_post = 0
  in_tag_count, wn_tag_count = {}, {}
  with open(dataset_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      tot_line += 1
      if (tot_line % 20000000) == 0:
        print('line#%09d' % (tot_line))

      fields = line.strip().split(sep_field)
      assert len(fields) == num_field
      if fields[idx_marker] != '0': # not image
        continue
      if len(fields[idx_tag]) == 0: # no tags
        continue
      user = fields[idx_user]
      if user not in user_set:
        continue
      is_valid = False
      tags = fields[idx_tag].split(sep_tag)
      for tag in tags:
        if tag in in_tag_set or tag in wn_tag_set:
          is_valid = True
          break
      if not is_valid:
        continue
      for tag in tags:
        if tag in in_tag_set:
          in_tag_count[tag] = in_tag_count.get(tag, 0) + 1
        if tag in wn_tag_set:
          wn_tag_count[tag] = wn_tag_count.get(tag, 0) + 1
      num_post += 1
  print('#user=%d #imagenet=%d #wordnet=%d #post=%d' % (
      len(user_set), len(in_tag_count), len(wn_tag_count), num_post))
  return in_tag_count, wn_tag_count, user_set

def main():
  in_tag_set = pickle.load(open(in_file_p, 'rb'))
  wn_tag_set = pickle.load(open(wn_file_p, 'rb'))

  while True:
    in_tag_count, wn_tag_count, user_set = reduce(in_tag_set, wn_tag_set)

    in_tag_set = set([t for t,c in in_tag_count.items() if c >= min_in_tag])
    wn_tag_set = set([t for t,c in wn_tag_count.items() if c >= min_wn_tag])

    if (min(in_tag_count.values()) >= min_in_tag and
        min(wn_tag_count.values()) >= min_wn_tag):
      break

  print('#imagenet=%d #wordnet=%d' % (len(in_tag_set), len(wn_tag_set)))
  pickle.dump(in_tag_set, open(in_file_f, 'wb'))
  pickle.dump(wn_tag_set, open(wn_file_f, 'wb'))
  pickle.dump(user_set, open(user_file_f, 'wb'))

if __name__ == '__main__':
  main()


