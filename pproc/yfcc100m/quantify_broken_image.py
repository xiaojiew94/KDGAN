from nltk.corpus import wordnet
from operator import itemgetter
from sys import stdout
from urllib import parse

dataset_file = '/data/yfcc100m/yfcc100m_dataset'
tag_file = '/data/yfcc100m/yfcc100m_tag'

n_field = 25
i_tags = 10
i_marker = 24

s_tags = ','


def main():
  print('quantitatively estimate the number of broken images')

  tag_count = {}
  t_line = 0
  with open(dataset_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break

      fields = line.strip().split('\t')
      assert len(fields) == n_field

      if fields[i_marker] != '0': # not image
        continue
      if len(fields[i_tags]) == 0: # no tags
        continue
      tags = fields[i_tags].split(s_tags)
      for tag in tags:
        tag = parse.unquote(tag)
        stdout.write('%s: %s\n' % (tag, ' -- '.join(tag.split())))
        input()
        tag_count[tag] = tag_count.get(tag, 0) + 1

      t_line += 1
      if (t_line % 5000000) == 0:
        print('line#%09d' % (t_line))
  print('%s contains %d lines in total' % (dataset_file, t_line))

  tag_count = sorted(tag_count.items(), key=itemgetter(1), reverse=True)
  with open(tag_file, 'w') as fout:
    for tag, count in tag_count:
      fout.write('%s\t%d\n' % (tag, count))

if __name__ == '__main__':
  main()


