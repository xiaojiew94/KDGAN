from nltk.corpus import wordnet
from operator import itemgetter
from sys import stdout

import pickle
import string

dataset_file = '/data/yfcc100m/yfcc100m_dataset'
tag_file = '/data/yfcc100m/yfcc100m_tag'
tmp_file = '/data/yfcc100m/yfcc100m_tmp'

sup_tag = 20

num_field = 25
idx_field = 10
idx_marker = 24

sep_tag = ','
sep_word = '+'

def main():
  unamb_nouns = pickle.load(open('unambiguous_noun.p', 'rb'))

  num_tag = 0
  tags = []
  with open(tag_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break

      fields = line.strip().split()
      tag, count = fields[0], int(fields[1])

      valid = tag in unamb_nouns
      if valid:
        num_tag += 1
        tags.append(tag)
  assert num_tag == len(tags)
  with open(tmp_file, 'w') as fout:
    for tag in tags:
      fout.write('%s\n' % (tag))
  print('%d valid tags' % (num_tag))

def test():
  tag_count = {}
  t_line = 0
  with open(dataset_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break

      fields = line.strip().split('\t')
      assert len(fields) == num_field

      if fields[idx_marker] != '0': # not image
        continue
      if len(fields[idx_field]) == 0: # no tags
        continue
      tags = fields[idx_field].split(sep_tag)
      for tag in tags:
        tag = tag.lower()
        words = tag.split(sep_word)
        valid = True
        for word in words:
          if not is_valid(word):
            valid = False
            break
        if valid:
          tag_count[tag] = tag_count.get(tag, 0) + 1

      t_line += 1
      # if t_line == 50000:
      #   break
      if (t_line % 5000000) == 0:
        print('line#%09d' % (t_line))
  print('%s contains %d lines in total' % (dataset_file, t_line))

  tag_count = sorted(tag_count.items(), key=itemgetter(1), reverse=True)
  with open(tag_file, 'w') as fout:
    for tag, count in tag_count:
      if count < sup_tag:
        break
      fout.write('%s\t%d\n' % (tag, count))

if __name__ == '__main__':
  main()


