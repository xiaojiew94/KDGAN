from utils import *

import utils

from nltk.corpus import wordnet
from os import path
from sys import stdout
from urllib import request

import argparse
import pickle
import pycountry
import re

import logging
logging.basicConfig(level=logging.INFO, format=log_format)

excluded_file = path.join(data_dir, 'in_excluded.txt')
def sort_excluded():
  imagenet_excl = set()
  with open(excluded_file) as fin:
    for line in fin.readlines():
      imagenet_excl.add(line.strip())
  with open(excluded_file, 'w') as fout:
    for excl in sorted(imagenet_excl):
      fout.write('%s\n' % (excl))
sort_excluded()

def get_imagenet_excl():
  imagenet_excl = set()

  for country in pycountry.countries:
    imagenet_excl.add(country.name.lower())
  for country in pycountry.historic_countries:
    imagenet_excl.add(country.name.lower())
  for subdivision in pycountry.subdivisions:
    imagenet_excl.add(subdivision.name.lower())

  with open(excluded_file) as fin:
    for line in fin.readlines():
      imagenet_excl.add(line.strip())

  return imagenet_excl

imagenet_excl = get_imagenet_excl()
def is_imagenet_tag(word):
  if word in imagenet_excl:
    return False
  if len(word) < 3:
    return False
  synsets = wordnet.synsets(word)
  if len(synsets) == 0:
    return False
  is_valid = False
  for synset in synsets:
    if synset.name().split('.')[1] == 'n':
      is_valid = True
      break
  return is_valid

def main():
  base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/'
  synset_url = '{}/imagenet_lsvrc_2015_synsets.txt'.format(base_url)
  synset_to_name_url = '{}/imagenet_metadata.txt'.format(base_url)
  filename, _ = request.urlretrieve(synset_url)
  synset_list = [s.strip() for s in open(filename).readlines()]
  num_synset = len(synset_list)
  assert num_synset == 1000
  filename, _ = request.urlretrieve(synset_to_name_url)
  synset_to_text_list = open(filename).readlines()
  tot_synset = len(synset_to_text_list)
  assert tot_synset == 21842
  synset_to_name = {}
  for s in synset_to_text_list:
    parts = s.strip().split('\t')
    assert len(parts) == 2
    synset = parts[0]
    text = parts[1]
    synset_to_name[synset] = text
  word_set = set()
  label_id = 0
  for synset in synset_list:
    text = synset_to_name[synset].lower()
    word_list = re.findall(r'[a-zA-Z-]+', text)
    label_id += 1
    word_set = word_set.union(word_list)
    print('#%04d %s' % (label_id, text))
    stdout.write('   ')
    for word in word_list:
      stdout.write(' %s' % (word))
    stdout.write('\n\n')

  imagenet_tag_set = set()
  for word in word_set:
    if is_imagenet_tag(word):
      imagenet_tag_set.add(word)
  utils.save_set_readable(imagenet_tag_set, in_all_noun_rfile)
  pickle.dump(imagenet_tag_set, open(in_all_noun_pfile, 'wb'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--override', action='store_true')
  args = parser.parse_args()
  if not path.isfile(in_all_noun_pfile) or args.override:
    main()
  else:
    logging.info('do not override')