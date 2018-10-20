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
      if (tot_line % 200000) == 0:
        logging.info('#line=%07d' % (tot_line))
  logging.info('#line=%7d' % (tot_line))
  print(num_fold)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('num_fold', type=int)
  args = parser.parse_args()
  main(args.num_fold)
