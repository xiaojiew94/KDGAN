from utils import *

import argparse
import os
import pickle

import logging
logging.basicConfig(level=logging.INFO, format=log_format)

def main(url_fold_file):
  with open(url_fold_file) as fin:
    for line in fin.readlines():
      image_url = line.strip()
      print(image_url)
      break

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('url_fold_file', type=str)
  args = parser.parse_args()
  main(args.url_fold_file)
