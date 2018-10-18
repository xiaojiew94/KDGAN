from operator import itemgetter
from urllib import request

num_field = 25
idx_user = 3
idx_tag = 10
idx_marker = 24
sep_field = '\t'
sep_tag = ','
in_file_p = 'imagenet_tag_set.p'
wn_file_p = 'wordnet_tag_set.p'
dataset_file = '/data/yfcc100m/yfcc100m_dataset'
def main():
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

  in_tag_set = pickle.load(open(in_file_p, 'rb'))
  wn_tag_set = pickle.load(open(wn_file_p, 'rb'))
  with open(dataset_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      tot_line += 1
      if tot_line >= 10000:
        break
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
      user_count[user] = user_count.get(user, 0) + 1
  with open('yfcc100m_top_user.txt', 'w') as fout:
    for user, count in user_count.items():
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