from utils import *

import utils

from os import path

import argparse
import os
import pickle
import requests

import logging
logging.basicConfig(level=logging.INFO, format=log_format)

def main(url_fold_file):
  image_fold_dir = url_fold_file.replace('url_', 'image_')
  print(image_fold_dir)
  with open(url_fold_file) as fin:
    for line in fin.readlines():
      image_url = line.strip()
      response = requests.get(image_url)
      print(image_url)
      print(response.url)
      print(response.is_redirect)
      input()
      continue
      image_file = utils.get_image_file(image_fold_dir, image_url)
      image_dir = path.dirname(image_file)
      if not path.exists(image_dir):
          os.makedirs(image_dir)
      # with open('/Users/scott/Downloads/cat3.jpg', 'wb') as f:  
      #     f.write(response.content)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('url_fold_file', type=str)
  args = parser.parse_args()
  main(args.url_fold_file)
