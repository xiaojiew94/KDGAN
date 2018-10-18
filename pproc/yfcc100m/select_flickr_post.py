from nltk.corpus import wordnet
from operator import itemgetter
from sys import stdout

import pickle
import string

dataset_file = '/data/yfcc100m/yfcc100m_dataset'
in_file_p = 'imagenet_tag_set.p'
wn_file_p = 'wordnet_tag_set.p'

num_field = 25
idx_field = 10
idx_marker = 24
sep_field = '\t'
sep_tag = ','

min_user = 20
min_in_tag = 20
min_wn_tag = 100

def get_tag_count(in_tag_set, wn_tag_set):
  with open(dataset_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      fields = line.strip().split(sep_field)
      assert len(fields) == num_field
      if fields[idx_marker] != '0': # not image
        continue
      is_valid = False
      tags = fields[idx_field].split(sep_tag)
      print(tags)
      input()
      for tag in tags:
        if tag in in_tag_set or tag in wn_tag_set:
          is_valid = True
          break


  in_tag_count, wn_tag_count = {}, {}
  return in_tag_count, wn_tag_count

def main():
  in_tag_set = pickle.load(open(in_file_p, 'rb'))
  wn_tag_set = pickle.load(open(wn_file_p, 'rb'))

  while True:
    print('#imagenet=%d #wordnet=%d' % (len(in_tag_set), len(wn_tag_set)))
    in_tag_count, wn_tag_count = get_tag_count(in_tag_set, wn_tag_set)

    in_tag_set = set([t for t, c in in_tag_count.items() if c >= min_in_tag])
    wn_tag_set = set([t for t, c in wn_tag_count.items() if c >= min_wn_tag])

    if (min(in_tag_count.values()) >= min_in_tag and
        min(wn_tag_count.values()) >= min_wn_tag):
      break
  print('#imagenet=%d #wordnet=%d' % (len(in_tag_set), len(wn_tag_set)))

if __name__ == '__main__':
  main()


