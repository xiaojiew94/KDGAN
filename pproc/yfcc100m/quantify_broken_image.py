dataset_infile = '/data/yfcc100m/yfcc100m_dataset'

def main():
  print('quantitatively estimate the number of broken images')

  t_line = 0
  with open(dataset_infile) as fin:
    while True:
      line = fin.readline()
      if not line:
        break

      fields = line.strip().split('\t')
      print(len(fields))
      exit()

      t_line += 1
      if (t_line % 1000000) == 0:
        print('line#%09d' % (t_line))
  print('%s contains %d lines in total' % (dataset_infile, t_line))

if __name__ == '__main__':
  main()