dataset_infile = '/data/yfcc100m/yfcc100m_dataset'
n_field = 25

i_tags = 10
i_marker = 24

s_tags = ','


def main():
  print('quantitatively estimate the number of broken images')

  t_line = 0
  with open(dataset_infile) as fin:
    while True:
      line = fin.readline()
      if not line:
        break

      fields = line.strip().split('\t')
      assert len(fields) == n_field

      if fields[i_marker] != '0':
        continue

      print(fields)

      user_tags = fields[i_tags].split(s_tags)
      input()

      t_line += 1
      if (t_line % 5000000) == 0:
        print('line#%09d' % (t_line))
  print('%s contains %d lines in total' % (dataset_infile, t_line))

if __name__ == '__main__':
  main()