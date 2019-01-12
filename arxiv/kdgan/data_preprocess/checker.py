from kdgan import config

from os import path

def check(dataset):
  dataset_dir = path.join(config.yfcc_dir, dataset)
  train_file = path.join(dataset_dir, '%s.train' % dataset)
  valid_file = path.join(dataset_dir, '%s.valid' % dataset)
  train_size = 0
  with open(train_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      fields = line.split('\t')
      assert len(fields) == 5
      for field in fields:
        assert len(field) > 0
        if len(field) == 0:
          print('%s has empty field' % (line))
      train_size += 1
  print('train size=%d' % (train_size))
  valid_size = 0
  with open(valid_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      fields = line.split('\t')
      assert len(fields) == 5
      for field in fields:
        assert len(field) > 0
        if len(field) == 0:
          print('%s has empty field' % (line))
      valid_size += 1
  print('valid size=%d' % (valid_size))

def main():
  check('yfcc10k')
  check('yfcc20k')

if __name__ == '__main__':
  main()