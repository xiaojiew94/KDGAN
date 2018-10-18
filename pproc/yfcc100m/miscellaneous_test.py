from operator import itemgetter
from urllib import request

import pickle

num_field = 25
idx_num_line = 0
idx_user = 3
idx_tag = 10
idx_marker = 24
sep_field = '\t'
sep_tag = ','
in_file_p = 'imagenet_tag_set.p'
wn_file_p = 'wordnet_tag_set.p'
dataset_file = '/data/yfcc100m/yfcc100m_dataset'
in_file_f = 'imagenet_tag_set.f'
wn_file_f = 'wordnet_tag_set.f'
yfcc_rnd_f = 'yfcc100m_rnd_tag.txt'
def main():
  in_tag_set = pickle.load(open(in_file_f, 'rb'))
  wn_tag_set = pickle.load(open(wn_file_f, 'rb'))
  with open(yfcc_rnd_f) as fin:
    for line in fin.readlines():
      fields = line.split()
      tag = fields[0]
      if tag not in in_tag_set and tag not in wn_tag_set:
        print(tag)
  return
  file = '/home/xiaojie/Projects/data/yfcc100m/yfcc_top/yfcc10k.data'
  user_set = set()
  with open(file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      fields = line.strip().split('\t')

      user = fields[1].upper()
      user_set.add(user)

  tot_line = 0
  user_count = {user:0 for user in user_set}
  in_tag_set = pickle.load(open(in_file_p, 'rb'))
  wn_tag_set = pickle.load(open(wn_file_p, 'rb'))
  # print('radio' in in_tag_set, 'radio' in wn_tag_set)
  tmp_tag_set = set()
  with open(dataset_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      tot_line += 1
      if (tot_line % 10000000) == 0:
        print(tmp_tag_set)
        print('line#%09d' % (tot_line))

      fields = line.strip().split(sep_field)
      assert len(fields) == num_field
      num_line = fields[idx_num_line]
      if num_line != '69578747':
        continue
      tags = fields[idx_tag].split(sep_tag)
      print(tags)
      input()
      if fields[idx_marker] != '0': # not image
        continue
      if len(fields[idx_tag]) == 0: # no tags
        continue
      user = fields[idx_user]
      if user not in user_set:
        continue
      is_valid = False
      for tag in tags:
        if tag in in_tag_set or tag in wn_tag_set:
          is_valid = True
          break
      if user == '40717756@N08':
        tmp_tag_set = tmp_tag_set.union(tags)
      if not is_valid:
        continue
      user_count[user] = user_count[user] + 1

  user_count = sorted(user_count.items(), key=itemgetter(1))
  with open('yfcc100m_top_user.txt', 'w') as fout:
    for user, count in user_count:
      fout.write('%s\t%d\n' % (user, count))

def test():
  file = '/home/xiaojie/Projects/data/yfcc100m/yfcc_rnd/yfcc_rnd.data'
  # file = '/home/xiaojie/Projects/data/yfcc100m/yfcc_top/yfcc10k.data'
  # print('miscellaneous test')
  tag_count = {}
  with open(file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break

      fields = line.strip().split('\t')
      tags = fields[-1].split()
      for tag in tags:
        tag_count[tag] = tag_count.get(tag, 0) + 1

  tag_count = sorted(tag_count.items(), key=itemgetter(1), reverse=True)
  for tag, count in tag_count:
    print('%s\t%d' % (tag, count))

if __name__ == '__main__':
  main()