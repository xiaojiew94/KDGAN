from utils import *

import argparse
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

  fold_size = tot_line // num_fold
  logging.info('#line=%d #fold=%d' % (tot_line, fold_size))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('num_fold', type=int)
  args = parser.parse_args()
  if not path.isdir(flickr_image_dir) or args.override:
    main(args.num_fold)
  else:
    logging.info('do not override')