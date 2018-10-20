from utils import *

import argparse
import os
import pickle

import logging
logging.basicConfig(level=logging.INFO, format=log_format)

def main(num_fold):
  tot_line = 0
  with open(yfcc_sample_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      tot_line += 1

  fold_size = tot_line // num_fold + 1
  logging.info('#line=%d #fold=%d' % (tot_line, fold_size))

  tot_line = 0
  os.makedirs(flickr_image_dir)
  with open(yfcc_sample_file) as fin:
    for fold in range(num_fold):
      url_fold_file = path.join(flickr_image_dir, 'url_fold%d' % fold)
      with open(url_fold_file, 'w') as fout:
        for _ in range(fold_size):
          line = fin.readline()
          if not line:
            logging.info('#line=%d break' % (tot_line))
            break
          tot_line += 1

          fields = line.strip().split(sep_field)
          image_url = fields[idx_image_url]
          print(image_url)
          input()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('num_fold', type=int)
  parser.add_argument('-o', '--override', action='store_true')
  args = parser.parse_args()
  if not path.isdir(flickr_image_dir) or args.override:
    main(args.num_fold)
  else:
    logging.info('do not override')