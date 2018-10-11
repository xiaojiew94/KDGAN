dataset_infile = '/data/yfcc100m/yfcc100m_dataset'
n_field = 25

i_tags = 10
i_marker = 24

s_tags = ','


def main():
  print('quantitatively estimate the number of broken images')

  tag_count = {}
  t_line = 0
  with open(dataset_infile) as fin:
    while True:
      line = fin.readline()
      if not line:
        break

      fields = line.strip().split('\t')
      assert len(fields) == n_field

      if fields[i_marker] != '0': # not image
        continue
      if len(s_tags) == 0: # no tags
        continue
      tags = fields[i_tags].split(s_tags)
      print(len(s_tags), tags)
      for tag in tags:
        tag_count[tag] = tag_count.get(tag, 0) + 1
      print(tag_count)
      input()

      t_line += 1
      if (t_line % 5000000) == 0:
        print('line#%09d' % (t_line))
  print('%s contains %d lines in total' % (dataset_infile, t_line))

if __name__ == '__main__':
  main()